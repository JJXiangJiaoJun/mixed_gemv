#pragma once

#include <type_traits>
#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

template<
 typename Mma_,
 typename AccumulatorCombine_,
 typename Epilogue_
>
struct MixedGemv {
public:

  using Mma = Mma_;
  using AccumulatorCombine = AccumulatorCombine_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Mma::LayoutC;
  using ElementScale = ElementC;

  static_assert(std::is_same_v<LayoutB, cutlass::layout::ColumnMajor>, "");

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreads = 32 * WarpCount::kCount;
  // static const bool kNeedBroadcastB = std::is_same_v<LayoutB, cutlass::layout::RowMajor>;

  static constexpr int kInterleave = Mma::IteratorA::Shape::kColumn / Mma::Shape::kK;

  static_assert(kInterleave == 4, "");

  struct Params {

    cutlass::gemm::GemmCoord problem_size;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB_LDG::Params params_B_LDG;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Mma::IteratorScale::Params params_scale;
    typename Mma::IteratorScale::TensorRef ref_scale;
    typename Mma::IteratorScale::Params params_zero_point;
    typename Mma::IteratorScale::TensorRef ref_zero_point;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;

    int group_size;
    int batch_count;

    int batch_stride_A;
    int batch_stride_B;
    int batch_stride_scale;
    int batch_stride_C;
    int batch_stride_D;

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(cutlass::gemm::GemmCoord const &problem_size_,
           typename Mma::IteratorA::TensorRef ref_A_,
           typename Mma::IteratorB::TensorRef ref_B_,
           typename Mma::IteratorScale::TensorRef ref_scale_,
           typename Mma::IteratorScale::TensorRef ref_zero_point_,
           typename Epilogue::OutputTileIterator::TensorRef ref_C_,
           typename Epilogue::OutputTileIterator::TensorRef ref_D_,
           typename OutputOp::Params output_op_ = typename OutputOp::Params(),
           int group_size_ = 64,
           int batch_count_ = 1,
           int batch_stride_A_ = 0,
           int batch_stride_B_ = 0,
           int batch_stride_scale_ = 0,
           int batch_stride_C_ = 0,
           int batch_stride_D_ = 0)
        : problem_size(problem_size_),
          params_A(ref_A_.layout()),
          ref_A(ref_A_),
          params_B(ref_B_.layout()),
          params_B_LDG(ref_B_.layout()),
          ref_B(ref_B_),
          params_scale(ref_scale_.layout()),
          ref_scale(ref_scale_),
          params_zero_point(ref_zero_point_.layout()),
          ref_zero_point(ref_zero_point_),
          params_C(ref_C_.layout()),
          ref_C(ref_C_),
          params_D(ref_D_.layout()),
          ref_D(ref_D_),
          output_op(output_op_),
          group_size(group_size_),
          batch_count(batch_count_),
          batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_),
          batch_stride_scale(batch_stride_scale_),
          batch_stride_C(batch_stride_C_),
          batch_stride_D(batch_stride_D_)
          {

          }
  };

  using SharedStorage = typename AccumulatorCombine::SharedStorage;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  MixedGemv() {}

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    cutlass::gemm::GemmCoord threadblock_tile_offset = cutlass::gemm::GemmCoord(blockIdx.x, 0, blockIdx.z);

    if (threadblock_tile_offset.m() * Mma::Shape::kM >= params.problem_size.m()) {
      return;
    }

    char *ptr_A = reinterpret_cast<char *>(params.ref_A.data());
    char *ptr_B = reinterpret_cast<char *>(params.ref_B.data());
    char *ptr_scale = reinterpret_cast<char *>(params.ref_scale.data());
    char *ptr_zero_point = reinterpret_cast<char *>(params.ref_zero_point.data());

    ///<
    ptr_A += threadblock_tile_offset.k() * params.batch_stride_A * cutlass::sizeof_bits<ElementA>::value / 8;
    ptr_B += threadblock_tile_offset.k() * params.batch_stride_B * cutlass::sizeof_bits<ElementB>::value / 8;
    ptr_scale += threadblock_tile_offset.k() * params.batch_stride_scale * cutlass::sizeof_bits<ElementScale>::value / 8;
    ptr_zero_point += threadblock_tile_offset.k() * params.batch_stride_scale * cutlass::sizeof_bits<ElementScale>::value / 8;

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM / kInterleave,
      0
    };

    cutlass::MatrixCoord tb_offset_B{
      0,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_scale{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      0
    };


    int gemm_k_iterations = (params.problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    int total_group = params.problem_size.k() / params.group_size;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A(
      params.params_A,
      reinterpret_cast<ElementA *>(ptr_A),
      {params.problem_size.m() / kInterleave, params.problem_size.k() * kInterleave},
      thread_idx,
      tb_offset_A
    );

    typename Mma::IteratorB iterator_B(
      params.params_B,
      reinterpret_cast<ElementB *>(ptr_B),
      {params.problem_size.k(), params.problem_size.m() * 4},
      thread_idx,
      tb_offset_B
    );

    typename Mma::IteratorB_LDG iterator_B_LDG(
      params.params_B_LDG,
      reinterpret_cast<ElementB *>(ptr_B),
      {params.problem_size.k(), params.problem_size.m()},
      thread_idx,
      tb_offset_B
    );

    typename Mma::IteratorScale iterator_scale(
      params.params_scale,
      reinterpret_cast<ElementScale *>(ptr_scale),
      {params.problem_size.m(), total_group},
      thread_idx,
      tb_offset_scale
    );

    typename Mma::IteratorScale iterator_zero_point(
      params.params_zero_point,
      reinterpret_cast<ElementScale *>(ptr_zero_point),
      {params.problem_size.m(), total_group},
      thread_idx,
      tb_offset_scale
    );

    Mma mma;

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    mma(gemm_k_iterations, params.group_size, accumulators, iterator_A, iterator_B, iterator_B_LDG, iterator_scale, iterator_zero_point, accumulators);

    ///< accumulator combine
    AccumulatorCombine acc_combine;
    acc_combine(shared_storage, accumulators);

    char *ptr_C = reinterpret_cast<char *>(params.ref_C.data());
    char *ptr_D = reinterpret_cast<char *>(params.ref_D.data());

    ptr_C += threadblock_tile_offset.k() * params.batch_stride_C * cutlass::sizeof_bits<ElementC>::value / 8;
    ptr_D += threadblock_tile_offset.k() * params.batch_stride_D * cutlass::sizeof_bits<ElementC>::value / 8;

    cutlass::MatrixCoord tb_offset_C{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      0
    };

    cutlass::MatrixCoord tb_offset_D{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      0
    };

    typename Epilogue::OutputTileIterator Iterator_C(
      params.params_C,
      reinterpret_cast<ElementC *>(ptr_C),
      {params.problem_size.m(), params.problem_size.n()},
      thread_idx,
      tb_offset_C
    );

    typename Epilogue::OutputTileIterator Iterator_D(
      params.params_D,
      reinterpret_cast<ElementC *>(ptr_D),
      {params.problem_size.m(), params.problem_size.n()},
      thread_idx,
      tb_offset_D
    );

    OutputOp output_op(params.output_op);
    Epilogue epilogue;
    epilogue(output_op, Iterator_D, accumulators, Iterator_C);
  }
};

}
}
}