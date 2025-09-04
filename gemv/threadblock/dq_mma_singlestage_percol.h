#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "gemv/threadblock/dq_mma_base.h"
#include "gemv/threadblock/dq_mma_singlestage.h"
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
class DqMmaSingleStageGemv<
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
 std::enable_if_t<!isFinegrained(QuantOp)>
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
>  {
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
  DqMmaSingleStageGemv() {}


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

    TransformAfterLDG ldg_converter;
    OperandBBroadcaster B_broadcaster;
    DqOperator dq_op;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;
    FragmentScale tb_frag_scale;
    FragmentZeroPoint tb_frag_zero;

    TransformFragmentA tb_transform_frag_A;
    BroadcastedFragmentB tb_broadcasted_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();
    tb_frag_scale.clear();
    tb_frag_zero.clear();

    iterator_A.load(tb_frag_A);
    ++iterator_A;

    iterator_B.load(tb_frag_B);
    ++iterator_B;

    iterator_scale.load(tb_frag_scale);
    ++iterator_scale;

    if constexpr (kHasZeroPoint) {
      iterator_zero_point.load(tb_frag_zero);
      ++iterator_zero_point;
    }

    tb_transform_frag_A = ldg_converter(tb_frag_A);
    tb_broadcasted_frag_B = B_broadcaster(tb_frag_B);

    WarpTileIteratorA warp_tile_iterator_A(tb_transform_frag_A);
    WarpTileIteratorB warp_tile_iterator_B(tb_broadcasted_frag_B);

    WarpTileFragmentA warp_tile_frag_A;
    WarpTileFragmentB warp_tile_frag_B;

    WarpTileFragmentA dq_warp_tile_frag_A;

    Operator warp_mma;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);
    iterator_scale.clear_mask(gemm_k_iterations <= 1);

    if constexpr (kHasZeroPoint) {
      iterator_zero_point.clear_mask(gemm_k_iterations <= 1);
    }

    for(; gemm_k_iterations > 0; --gemm_k_iterations) {

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

        warp_tile_iterator_A.load(warp_tile_frag_A);
        warp_tile_iterator_B.load(warp_tile_frag_B);

        ///< permform dq here
        dq_warp_tile_frag_A = dq_op(warp_tile_frag_A, tb_frag_scale, tb_frag_zero);

        warp_mma(accum, dq_warp_tile_frag_A, warp_tile_frag_B, accum);

        ++warp_tile_iterator_A;
        ++warp_tile_iterator_B;
      }

      warp_tile_iterator_A.reset();
      warp_tile_iterator_B.reset();

      iterator_A.load(tb_frag_A);
      ++iterator_A;

      iterator_B.load(tb_frag_B);
      ++iterator_B;

      tb_transform_frag_A = ldg_converter(tb_frag_A);
      tb_broadcasted_frag_B = B_broadcaster(tb_frag_B);

      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 2);
      iterator_B.clear_mask(gemm_k_iterations <= 2);
    }
  }

};

}
}
}