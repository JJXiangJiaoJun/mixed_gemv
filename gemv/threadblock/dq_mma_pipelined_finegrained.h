#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "gemv/threadblock/dq_mma_base.h"
#include "gemv/threadblock/dq_mma_pipelined.h"
#include "gemv/warp/warp_dequantizer.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Iterates over tiles of A operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of scale operand in global memory
  typename IteratorScale_,
  /// Iterates over tiles of scale operand in global memory
  typename IteratorZeroPoint_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  /// Converter for A matrix applied immediately after the LDG (before STS)
  typename TransformAfterLDG_,
  /// The quantization operator being used
  WeightOnlyQuantOp QuantOp
>
class DqMmaPipelinedGemv<
 Shape_,
 IteratorA_,
 IteratorB_,
 IteratorScale_,
 IteratorZeroPoint_,
 ElementC_,
 LayoutC_,
 Policy_,
 TransformAfterLDG_,
 QuantOp,
 std::enable_if_t<isFinegrained(QuantOp)>
> : public DqMmaBase<
 Shape_,
 IteratorA_,
 IteratorB_,
 IteratorScale_,
 IteratorZeroPoint_,
 ElementC_,
 LayoutC_,
 Policy_,
 TransformAfterLDG_
> {
public:

  using Base = DqMmaBase<
                        Shape_,
                        IteratorA_,
                        IteratorB_,
                        IteratorScale_,
                        IteratorZeroPoint_,
                        ElementC_,
                        LayoutC_,
                        Policy_,
                        TransformAfterLDG_>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using IteratorScale = IteratorScale_; ///< Iterates over tiles of Scale operand in global memory
  using IteratorZeroPoint = IteratorZeroPoint_; ///< Iterates over tiles of Zero-point operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details
  using TransformAfterLDG = TransformAfterLDG_;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using FragmentScale = typename IteratorScale::Fragment;
  using FragmentZeroPoint = typename IteratorZeroPoint::Fragment;

  using TransformFragmentA = typename TransformAfterLDG::result_type;
  using OperandBBroadcaster = typename Base::OperandBBroadcaster;
  using BroadcastedFragmentB = typename Base::BroadcastedFragmentB;

  /// Fragment of accumulator tile
  using FragmentC = typename Operator::FragmentC;


  using WarpTileIteratorA = typename Operator::IteratorA;
  using WarpTileIteratorB = typename Operator::IteratorB;

  using WarpTileFragmentA = typename Operator::FragmentA;
  using WarpTileFragmentB = typename Operator::FragmentB;

  static_assert(WarpTileFragmentA::kElements == FragmentScale::kElements, "");


  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;

  static const int kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::WarpShape::kColumn / Operator::InstructionShape::kK);

  static const WeightOnlyQuantOp kQuantOp = QuantOp;

  static const bool kHasZeroPoint = hasZero(kQuantOp);

  using DqOperator = cutlass::gemm::warp::MmaOpDequantizer<Operator, typename WarpTileFragmentA::Element, kQuantOp, WarpTileFragmentA::kElements>;

  /// Number of stages
  static int const kStages = 1;

  struct SharedStorage {};


public:
  CUTLASS_HOST_DEVICE
  DqMmaPipelinedGemv() {}


  CUTLASS_DEVICE
  void load_scales_zeros_and_advance(IteratorScale& iterator_scale, IteratorZeroPoint& iterator_zero_point,
                                     FragmentScale& frag_scale, FragmentZeroPoint& frag_zero, int &row_groupsize64, int group_size) {

    iterator_scale.load(frag_scale);
    ///< advance iterator scale
    if (group_size == 64) {
      ++iterator_scale;
    } else if (group_size == 128) {
      if constexpr (Shape::kK == 128) {
        ++iterator_scale;
      }
      else if constexpr (Shape::kK == 64) {
        if (row_groupsize64 & 0x1)
        {
          ++iterator_scale;
        }
      }
      else
      {
          static_assert(Shape::kK == 0, "Unsupported k tile shape, can only be 64 or 128");
      }
    }

    if constexpr (kHasZeroPoint) {
      iterator_zero_point.load(frag_zero);
      ///< advance iterator scale
      if (group_size == 64) {
        ++iterator_zero_point;
      } else if (group_size == 128) {
        if constexpr (Shape::kK == 128) {
          ++iterator_zero_point;
        }
        else if constexpr (Shape::kK == 64) {
          if (row_groupsize64 & 0x1)
          {
            ++iterator_zero_point;
          }
        }
        else
        {
            static_assert(Shape::kK == 0, "Unsupported k tile shape, can only be 64 or 128");
        }
      }
    }

    ++row_groupsize64;
  }




  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,         ///< number of iterations of the mainloop
    int group_size,
    FragmentC &accum,              ///< destination accumulator tile
    IteratorA iterator_A,          ///< iterator over A operand in global memory
    IteratorB iterator_B,          ///< iterator over B operand in global memory
    IteratorScale iterator_scale,  ///< iterator over Scale operand in global memory
    IteratorZeroPoint iterator_zero_point,  ///< iterator over Scale operand in global memory
    FragmentC const &src_accum) {  ///< source accumualtor tile

    accum = src_accum;

    int row_groupsize64 = 0;

    TransformAfterLDG ldg_converter;
    OperandBBroadcaster B_broadcaster;
    DqOperator dq_op;

    FragmentA tb_frag_A[2];
    FragmentB tb_frag_B[2];
    FragmentScale tb_frag_scale[2];
    FragmentZeroPoint tb_frag_zero[2];

    TransformFragmentA tb_transform_frag_A[2];
    BroadcastedFragmentB tb_broadcasted_frag_B[2];

    tb_frag_A[0].clear();
    tb_frag_A[1].clear();
    tb_frag_B[0].clear();
    tb_frag_B[1].clear();
    tb_frag_scale[0].clear();
    tb_frag_scale[1].clear();
    tb_frag_zero[0].clear();
    tb_frag_zero[1].clear();

    iterator_A.load(tb_frag_A[0]);
    ++iterator_A;

    iterator_B.load(tb_frag_B[0]);
    ++iterator_B;

    load_scales_zeros_and_advance(iterator_scale, iterator_zero_point, tb_frag_scale[0], tb_frag_zero[0], row_groupsize64, group_size);

    WarpTileIteratorA warp_tile_iterator_A[2];
    WarpTileIteratorB warp_tile_iterator_B[2];

    warp_tile_iterator_A[0].reset(tb_transform_frag_A[0]);
    warp_tile_iterator_A[1].reset(tb_transform_frag_A[1]);

    warp_tile_iterator_B[0].reset(tb_broadcasted_frag_B[0]);
    warp_tile_iterator_B[1].reset(tb_broadcasted_frag_B[1]);

    WarpTileFragmentA warp_tile_frag_A[2];
    WarpTileFragmentB warp_tile_frag_B[2];

    WarpTileFragmentA dq_warp_tile_frag_A[2];

    Operator warp_mma;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);
    iterator_scale.clear_mask(gemm_k_iterations <= 1);

    if constexpr (kHasZeroPoint) {
      iterator_zero_point.clear_mask(gemm_k_iterations <= 1);
    }

    for(; gemm_k_iterations > 0;) {

      ///< Load next buffer
      iterator_A.load(tb_frag_A[1]);
      ++iterator_A;

      iterator_B.load(tb_frag_B[1]);
      ++iterator_B;

      load_scales_zeros_and_advance(iterator_scale, iterator_zero_point, tb_frag_scale[1], tb_frag_zero[1], row_groupsize64, group_size);


      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 2);
      iterator_B.clear_mask(gemm_k_iterations <= 2);
      iterator_scale.clear_mask(gemm_k_iterations <= 2);

      if constexpr (kHasZeroPoint) {
        iterator_zero_point.clear_mask(gemm_k_iterations <= 2);
      }

      ///< Mma stage 0
      tb_transform_frag_A[0] = ldg_converter(tb_frag_A[0]);
      tb_broadcasted_frag_B[0] = B_broadcaster(tb_frag_B[0]);

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

        warp_tile_iterator_A[0].load(warp_tile_frag_A[0]);
        warp_tile_iterator_B[0].load(warp_tile_frag_B[0]);

        ///< permform dq here
        dq_warp_tile_frag_A[0] = dq_op(warp_tile_frag_A[0], tb_frag_scale[0], tb_frag_zero[0]);

        warp_mma(accum, dq_warp_tile_frag_A[0], warp_tile_frag_B[0], accum);

        ++warp_tile_iterator_A[0];
        ++warp_tile_iterator_B[0];
      }

      warp_tile_iterator_A[0].reset();
      warp_tile_iterator_B[0].reset();
      --gemm_k_iterations;

      ///< Advance next stage
      iterator_A.load(tb_frag_A[0]);
      ++iterator_A;

      iterator_B.load(tb_frag_B[0]);
      ++iterator_B;

      load_scales_zeros_and_advance(iterator_scale, iterator_zero_point, tb_frag_scale[0], tb_frag_zero[0], row_groupsize64, group_size);

      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 2);
      iterator_B.clear_mask(gemm_k_iterations <= 2);
      iterator_scale.clear_mask(gemm_k_iterations <= 2);
      if constexpr (kHasZeroPoint) {
       iterator_zero_point.clear_mask(gemm_k_iterations <= 2);
      }


      if (gemm_k_iterations > 0) {

        ///< Mma stage 1
        tb_transform_frag_A[1] = ldg_converter(tb_frag_A[1]);
        tb_broadcasted_frag_B[1] = B_broadcaster(tb_frag_B[1]);

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

          warp_tile_iterator_A[1].load(warp_tile_frag_A[1]);
          warp_tile_iterator_B[1].load(warp_tile_frag_B[1]);

          ///< permform dq here
          dq_warp_tile_frag_A[1] = dq_op(warp_tile_frag_A[1], tb_frag_scale[1], tb_frag_zero[1]);

          warp_mma(accum, dq_warp_tile_frag_A[1], warp_tile_frag_B[1], accum);

          ++warp_tile_iterator_A[1];
          ++warp_tile_iterator_B[1];
        }

        warp_tile_iterator_A[1].reset();
        warp_tile_iterator_B[1].reset();
      }

      --gemm_k_iterations;

    }
  }

};

}
}
}