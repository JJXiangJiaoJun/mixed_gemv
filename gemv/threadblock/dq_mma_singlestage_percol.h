#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "gemv/threadblock/dq_mma_singlestage.h"
#include "gemv/warp/warp_dequantizer.h"

// #define DOUBLE_BUFFER

// #define PRINT

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
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_LDG_,
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
 IteratorB_LDG_,
 IteratorScale_,
 IteratorZeroPoint_,
 ElementC_,
 LayoutC_,
 Policy_,
 TransformAfterLDG_,
 QuantOp,
 std::enable_if_t<!isFinegrained(QuantOp)>
> {
public:
  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using IteratorB_LDG = IteratorB_LDG_;     ///< Iterates over tiles of B operand in global memory
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
  using FragmentB_LDG = typename IteratorB_LDG::Fragment;

  static const int LDG_Factor = FragmentB::kElements / FragmentB_LDG::kElements;

  using FragmentScale = typename IteratorScale::Fragment;
  using FragmentZeroPoint = typename IteratorZeroPoint::Fragment;

  using TransformFragmentA = typename TransformAfterLDG::result_type;

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

#ifdef DOUBLE_BUFFER

  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,         ///< number of iterations of the mainloop
    int group_size,
    FragmentC &accum,              ///< destination accumulator tile
    IteratorA iterator_A,          ///< iterator over A operand in global memory
    IteratorB iterator_B,          ///< iterator over B operand in global memory
    IteratorB_LDG iterator_B_LDG,          ///< iterator over B operand in global memory
    IteratorScale iterator_scale,  ///< iterator over Scale operand in global memory
    IteratorZeroPoint iterator_zero_point,  ///< iterator over Scale operand in global memory
    FragmentC const &src_accum) {  ///< source accumualtor tile

    accum = src_accum;

    TransformAfterLDG ldg_converter;
    DqOperator dq_op;

    FragmentA tb_frag_A[2];
    FragmentB_LDG tb_frag_B_ldg[2];
    FragmentB tb_frag_B[2];
    FragmentScale tb_frag_scale;
    FragmentZeroPoint tb_frag_zero;

    TransformFragmentA tb_transform_frag_A[2];

    tb_frag_A[0].clear();
    tb_frag_A[1].clear();
    tb_frag_B_ldg[0].clear();
    tb_frag_B_ldg[1].clear();
    tb_frag_B[0].clear();
    tb_frag_B[1].clear();
    tb_frag_scale.clear();
    tb_frag_zero.clear();

    iterator_scale.load(tb_frag_scale);
    ++iterator_scale;

    if constexpr (kHasZeroPoint) {
      iterator_zero_point.load(tb_frag_zero);
      ++iterator_zero_point;
    }


    iterator_A.load(tb_frag_A[0]);
    ++iterator_A;

    iterator_B_LDG.load(tb_frag_B_ldg[0]);
    ++iterator_B_LDG;

    // iterator_B.load(tb_frag_B[0]);
    // ++iterator_B;

    FragmentB_LDG *tb_frag_B_ptr[2];

    tb_frag_B_ptr[0] = reinterpret_cast<FragmentB_LDG *>(&tb_frag_B[0]);
    tb_frag_B_ptr[1] = reinterpret_cast<FragmentB_LDG *>(&tb_frag_B[1]);

    iterator_A.clear_mask(gemm_k_iterations <= 1);
    // iterator_B.clear_mask(gemm_k_iterations <= 1);
    iterator_B_LDG.clear_mask(gemm_k_iterations <= 1);

    iterator_A.load(tb_frag_A[1]);
    ++iterator_A;

    // iterator_B.load(tb_frag_B[1]);
    // ++iterator_B;

    iterator_B_LDG.load(tb_frag_B_ldg[1]);
    ++iterator_B_LDG;

    // WarpTileIteratorA warp_tile_iterator_A(tb_transform_frag_A);
    // WarpTileIteratorB warp_tile_iterator_B(tb_frag_B);

    // WarpTileFragmentA warp_tile_frag_A;
    // WarpTileFragmentB warp_tile_frag_B;

    // WarpTileFragmentA dq_warp_tile_frag_A;

    Operator warp_mma;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 2);
    // iterator_B.clear_mask(gemm_k_iterations <= 2);
    iterator_B_LDG.clear_mask(gemm_k_iterations <= 2);

    for(; gemm_k_iterations > 0;) {


      {

        tb_transform_frag_A[0] = ldg_converter(tb_frag_A[0]);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < LDG_Factor; ++i) {
          *(tb_frag_B_ptr[0] + i) = tb_frag_B_ldg[0];
        }

        WarpTileIteratorA warp_tile_iterator_A(tb_transform_frag_A[0]);
        WarpTileIteratorB warp_tile_iterator_B(tb_frag_B[0]);

        WarpTileFragmentA warp_tile_frag_A;
        WarpTileFragmentB warp_tile_frag_B;
        WarpTileFragmentA dq_warp_tile_frag_A;

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
          warp_tile_iterator_A.load(warp_tile_frag_A);
          warp_tile_iterator_B.load(warp_tile_frag_B);

          dq_warp_tile_frag_A = dq_op(warp_tile_frag_A, tb_frag_scale, tb_frag_zero);
          warp_mma(accum, dq_warp_tile_frag_A, warp_tile_frag_B, accum);

          ++warp_tile_iterator_A;
          ++warp_tile_iterator_B;
        }
      }


        // 预取下一组数据到缓冲区0（如果还有剩余迭代）
      iterator_A.load(tb_frag_A[0]);
      ++iterator_A;

      iterator_B_LDG.load(tb_frag_B_ldg[0]);
      ++iterator_B_LDG;

      // CUTLASS_PRAGMA_UNROLL
      // for (int i = 0; i < LDG_Factor; ++i) {
      //   *(tb_frag_B_ptr[0] + i) = tb_frag_B_ldg[0];
      // }

      // iterator_B.load(tb_frag_B[0]);
      // ++iterator_B;

      // tb_transform_frag_A[0] = ldg_converter(tb_frag_A[0]);
      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 3);
      // iterator_B.clear_mask(gemm_k_iterations <= 3);
      iterator_B_LDG.clear_mask(gemm_k_iterations <= 3);
      --gemm_k_iterations;

#ifdef PRINT

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("gemm_k_iterations %d, threadIdx %d, tb_transform_frag_A[0]:{"
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
            "\n"
             , gemm_k_iterations,
             threadIdx.x,
             float(*(tb_transform_frag_A[0].data() + 0)), float(*(tb_transform_frag_A[0].data() + 1)), float(*(tb_transform_frag_A[0].data() + 2)), float(*(tb_transform_frag_A[0].data() + 3)),
             float(*(tb_transform_frag_A[0].data() + 4)), float(*(tb_transform_frag_A[0].data() + 5)), float(*(tb_transform_frag_A[0].data() + 6)), float(*(tb_transform_frag_A[0].data() + 7)),
             float(*(tb_transform_frag_A[0].data() + 8)), float(*(tb_transform_frag_A[0].data() + 9)), float(*(tb_transform_frag_A[0].data() + 10)), float(*(tb_transform_frag_A[0].data() + 11)),
             float(*(tb_transform_frag_A[0].data() + 12)), float(*(tb_transform_frag_A[0].data() + 13)), float(*(tb_transform_frag_A[0].data() + 14)), float(*(tb_transform_frag_A[0].data() + 15))

             );


      printf("gemm_k_iterations %d, threadIdx %d, tb_frag_B[0]0-15:{"
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
            "}\n"
             , gemm_k_iterations,
             threadIdx.x,
             float(*(tb_frag_B[0].data() + 0)),  float(*(tb_frag_B[0].data() + 1)),  float(*(tb_frag_B[0].data() + 2)),  float(*(tb_frag_B[0].data() + 3)),
             float(*(tb_frag_B[0].data() + 4)),  float(*(tb_frag_B[0].data() + 5)),  float(*(tb_frag_B[0].data() + 6)),  float(*(tb_frag_B[0].data() + 7)),
             float(*(tb_frag_B[0].data() + 8)),  float(*(tb_frag_B[0].data() + 9)),  float(*(tb_frag_B[0].data() + 10)), float(*(tb_frag_B[0].data() + 11)),
             float(*(tb_frag_B[0].data() + 12)), float(*(tb_frag_B[0].data() + 13)), float(*(tb_frag_B[0].data() + 14)), float(*(tb_frag_B[0].data() + 15))

             );

      printf("gemm_k_iterations %d, threadIdx %d, tb_frag_B[0]16-31:{"
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
            "}\n"
             ,  gemm_k_iterations,
             threadIdx.x,
             float(*(tb_frag_B[0].data() + 16)),  float(*(tb_frag_B[0].data() + 17)),  float(*(tb_frag_B[0].data() + 18)),  float(*(tb_frag_B[0].data() + 19)),
             float(*(tb_frag_B[0].data() + 20)),  float(*(tb_frag_B[0].data() + 21)),  float(*(tb_frag_B[0].data() + 22)),  float(*(tb_frag_B[0].data() + 23)),
             float(*(tb_frag_B[0].data() + 24)),  float(*(tb_frag_B[0].data() + 25)),  float(*(tb_frag_B[0].data() + 26)),  float(*(tb_frag_B[0].data() + 27)),
             float(*(tb_frag_B[0].data() + 28)),  float(*(tb_frag_B[0].data() + 29)),  float(*(tb_frag_B[0].data() + 30)),  float(*(tb_frag_B[0].data() + 31))

             );
      printf("\n\n");
    }

#endif
      if (gemm_k_iterations > 0)
      {

        tb_transform_frag_A[1] = ldg_converter(tb_frag_A[1]);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < LDG_Factor; ++i) {
          *(tb_frag_B_ptr[1] + i) = tb_frag_B_ldg[1];
        }

        WarpTileIteratorA warp_tile_iterator_A(tb_transform_frag_A[1]);
        WarpTileIteratorB warp_tile_iterator_B(tb_frag_B[1]);

        WarpTileFragmentA warp_tile_frag_A;
        WarpTileFragmentB warp_tile_frag_B;
        WarpTileFragmentA dq_warp_tile_frag_A;

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
          warp_tile_iterator_A.load(warp_tile_frag_A);
          warp_tile_iterator_B.load(warp_tile_frag_B);

          dq_warp_tile_frag_A = dq_op(warp_tile_frag_A, tb_frag_scale, tb_frag_zero);
          warp_mma(accum, dq_warp_tile_frag_A, warp_tile_frag_B, accum);

          ++warp_tile_iterator_A;
          ++warp_tile_iterator_B;
        }
      }

      iterator_A.load(tb_frag_A[1]);
      ++iterator_A;

      iterator_B_LDG.load(tb_frag_B_ldg[1]);
      ++iterator_B_LDG;

      // CUTLASS_PRAGMA_UNROLL
      // for (int i = 0; i < LDG_Factor; ++i) {
      //   *(tb_frag_B_ptr[1] + i) = tb_frag_B_ldg[1];
      // }

      // iterator_B.load(tb_frag_B[1]);
      // ++iterator_B;


      // tb_transform_frag_A[1] = ldg_converter(tb_frag_A[1]);
      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 3);
      // iterator_B.clear_mask(gemm_k_iterations <= 3);
      iterator_B_LDG.clear_mask(gemm_k_iterations <= 3);
      --gemm_k_iterations;


#ifdef PRINT

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("gemm_k_iterations %d, threadIdx %d, tb_transform_frag_A[1]:{"
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
            "\n"
             , gemm_k_iterations,
             threadIdx.x,
             float(*(tb_transform_frag_A[1].data() + 0)),  float(*(tb_transform_frag_A[1].data() + 1)),  float(*(tb_transform_frag_A[1].data() + 2)),  float(*(tb_transform_frag_A[1].data() + 3)),
             float(*(tb_transform_frag_A[1].data() + 4)),  float(*(tb_transform_frag_A[1].data() + 5)),  float(*(tb_transform_frag_A[1].data() + 6)),  float(*(tb_transform_frag_A[1].data() + 7)),
             float(*(tb_transform_frag_A[1].data() + 8)),  float(*(tb_transform_frag_A[1].data() + 9)),  float(*(tb_transform_frag_A[1].data() + 10)), float(*(tb_transform_frag_A[1].data() + 11)),
             float(*(tb_transform_frag_A[1].data() + 12)), float(*(tb_transform_frag_A[1].data() + 13)), float(*(tb_transform_frag_A[1].data() + 14)), float(*(tb_transform_frag_A[1].data() + 15))

             );

      printf("gemm_k_iterations %d, threadIdx %d, tb_frag_B[1]0-15:{"
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
            "}\n"
             , gemm_k_iterations,
             threadIdx.x,
             float(*(tb_frag_B[1].data() + 0)),  float(*(tb_frag_B[1].data() + 1)),  float(*(tb_frag_B[1].data() + 2)),  float(*(tb_frag_B[1].data() + 3)),
             float(*(tb_frag_B[1].data() + 4)),  float(*(tb_frag_B[1].data() + 5)),  float(*(tb_frag_B[1].data() + 6)),  float(*(tb_frag_B[1].data() + 7)),
             float(*(tb_frag_B[1].data() + 8)),  float(*(tb_frag_B[1].data() + 9)),  float(*(tb_frag_B[1].data() + 10)), float(*(tb_frag_B[1].data() + 11)),
             float(*(tb_frag_B[1].data() + 12)), float(*(tb_frag_B[1].data() + 13)), float(*(tb_frag_B[1].data() + 14)), float(*(tb_frag_B[1].data() + 15))

             );

      printf("gemm_k_iterations %d, threadIdx %d, tb_frag_B[1]16-31:{"
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
             "%f, %f, %f, %f, "
            "}\n"
             , gemm_k_iterations,
             threadIdx.x,
             float(*(tb_frag_B[1].data() + 16)),  float(*(tb_frag_B[1].data() + 17)),  float(*(tb_frag_B[1].data() + 18)),  float(*(tb_frag_B[1].data() + 19)),
             float(*(tb_frag_B[1].data() + 20)),  float(*(tb_frag_B[1].data() + 21)),  float(*(tb_frag_B[1].data() + 22)),  float(*(tb_frag_B[1].data() + 23)),
             float(*(tb_frag_B[1].data() + 24)),  float(*(tb_frag_B[1].data() + 25)),  float(*(tb_frag_B[1].data() + 26)),  float(*(tb_frag_B[1].data() + 27)),
             float(*(tb_frag_B[1].data() + 28)),  float(*(tb_frag_B[1].data() + 29)),  float(*(tb_frag_B[1].data() + 30)),  float(*(tb_frag_B[1].data() + 31))

             );
      printf("\n\n");
    }
#endif

    }

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("threadIdx %d, accum: {%f, %f, %f, %f}\n", threadIdx.x,
    //          float(accum[0]), float(accum[1]), float(accum[2]), float(accum[3]));

    // }

  }


#else

  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,         ///< number of iterations of the mainloop
    int group_size,
    FragmentC &accum,              ///< destination accumulator tile
    IteratorA iterator_A,          ///< iterator over A operand in global memory
    IteratorB iterator_B,          ///< iterator over B operand in global memory
    IteratorB_LDG iterator_B_LDG,          ///< iterator over B operand in global memory
    IteratorScale iterator_scale,  ///< iterator over Scale operand in global memory
    IteratorZeroPoint iterator_zero_point,  ///< iterator over Scale operand in global memory
    FragmentC const &src_accum) {  ///< source accumualtor tile

    accum = src_accum;

    TransformAfterLDG ldg_converter;
    DqOperator dq_op;

    FragmentA tb_frag_A;
    FragmentB_LDG tb_frag_B_ldg;
    FragmentB tb_frag_B;
    FragmentScale tb_frag_scale;
    FragmentZeroPoint tb_frag_zero;

    TransformFragmentA tb_transform_frag_A;

    tb_frag_A.clear();
    tb_frag_B_ldg.clear();
    tb_frag_B.clear();
    tb_frag_scale.clear();
    tb_frag_zero.clear();

    iterator_A.load(tb_frag_A);
    ++iterator_A;

    // iterator_B.load(tb_frag_B);
    // ++iterator_B;

    iterator_B_LDG.load(tb_frag_B_ldg);
    ++iterator_B_LDG;

    FragmentB_LDG *tb_frag_B_ptr = reinterpret_cast<FragmentB_LDG *>(&tb_frag_B);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LDG_Factor; ++i) {
      *(tb_frag_B_ptr + i) = tb_frag_B_ldg;
    }

    iterator_scale.load(tb_frag_scale);
    ++iterator_scale;

    if constexpr (kHasZeroPoint) {
      iterator_zero_point.load(tb_frag_zero);
      ++iterator_zero_point;
    }

    tb_transform_frag_A = ldg_converter(tb_frag_A);

    WarpTileIteratorA warp_tile_iterator_A(tb_transform_frag_A);
    WarpTileIteratorB warp_tile_iterator_B(tb_frag_B);

    WarpTileFragmentA warp_tile_frag_A;
    WarpTileFragmentB warp_tile_frag_B;

    WarpTileFragmentA dq_warp_tile_frag_A;

    Operator warp_mma;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B_LDG.clear_mask(gemm_k_iterations <= 1);
    // iterator_B.clear_mask(gemm_k_iterations <= 1);
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

      // iterator_B.load(tb_frag_B);
      // ++iterator_B;


      iterator_B_LDG.load(tb_frag_B_ldg);
      ++iterator_B_LDG;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < LDG_Factor; ++i) {
        *(tb_frag_B_ptr + i) = tb_frag_B_ldg;
      }

      tb_transform_frag_A = ldg_converter(tb_frag_A);

      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 2);
      // iterator_B.clear_mask(gemm_k_iterations <= 2);
      iterator_B_LDG.clear_mask(gemm_k_iterations <= 2);
    }
  }

#endif

};

}
}
}