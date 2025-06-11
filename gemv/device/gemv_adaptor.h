#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/device_kernel.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemv/kernel/default_mixed_gemv.h"

namespace cutlass {
namespace gemm {
namespace device {

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for the input scale
  typename ElementScale_,
  /// Layout for the scale operand
  typename LayoutScale_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Operation performed by GEMV
  typename Operator_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadBlockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Instruction-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement_,
  /// Number of stages used in the pipelined mainloop
  typename EpilogueOutputOp_,
  int Stages,
  int AlignmentA,
  int AlignmentB,
  int AlignmentScale
>
class GemvAdaptor {
public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementScale = ElementScale_;
  using LayoutScale = LayoutScale_;
  using TensorRefScale = TensorRef<ElementScale const, LayoutScale>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using Operator = Operator_;
  using ThreadBlockShape = ThreadBlockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using EpilogueOutputOp = EpilogueOutputOp_;

  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentScale = AlignmentScale;


  using GemvKernel = typename cutlass::gemm::kernel::DefaultMixedGemv<ElementA,
                                                                      LayoutA,
                                                                      kAlignmentA,
                                                                      ElementB,
                                                                      LayoutB,
                                                                      kAlignmentB,
                                                                      ElementScale,
                                                                      LayoutScale,
                                                                      kAlignmentScale,
                                                                      ElementC,
                                                                      LayoutC,
                                                                      ElementAccumulator,
                                                                      OperatorClass,
                                                                      ArchTag,
                                                                      Operator,
                                                                      ThreadBlockShape,
                                                                      WarpShape,
                                                                      InstructionShape,
                                                                      WarpThreadArrangement,
                                                                      EpilogueOutputOp>::Kernel;

  using MmaLayoutA = typename GemvKernel::Mma::IteratorA::Layout;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, MmaLayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementScale const, LayoutScale> ref_scale;
    TensorRef<ElementScale const, LayoutScale> ref_zero_point;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int group_size;
    int batch_count;

    int batch_stride_A;
    int batch_stride_B;
    int batch_stride_scale;
    int batch_stride_C;
    int batch_stride_D;

    //
    // Methods
    //

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, MmaLayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementScale const, LayoutScale> ref_scale_,
      TensorRef<ElementScale const, LayoutScale> ref_zero_point_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int group_size_ = 64,
      int batch_count_ = 1,
      int batch_stride_A_ = 0,
      int batch_stride_B_ = 0,
      int batch_stride_scale_ = 0,
      int batch_stride_C_ = 0,
      int batch_stride_D_ = 0
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_scale(ref_scale_),
      ref_zero_point(ref_zero_point_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      group_size(group_size_),
      batch_count(batch_count_),
      batch_stride_A(batch_stride_A_),
      batch_stride_B(batch_stride_B_),
      batch_stride_scale(batch_stride_scale_),
      batch_stride_C(batch_stride_C_),
      batch_stride_D(batch_stride_D_) {

      }
  };

private:

  /// Kernel parameters object
  typename GemvKernel::Params params_;

public:

  CUTLASS_HOST_DEVICE
  GemvAdaptor() {}

  static dim3 get_grid_size(typename GemvKernel::Params &params) {
    return dim3((params.problem_size.m() + GemvKernel::Mma::Shape::kM - 1) / GemvKernel::Mma::Shape::kM, 1, params.batch_count);
  }

  static dim3 get_block_size(typename GemvKernel::Params &params) {
    return dim3(GemvKernel::kThreads, 1, 1);
  }

  void initialize(Arguments const &args, cudaStream_t stream = nullptr) {
    if (args.group_size < ThreadBlockShape::kK) {
      std::cout << "Group size must be large than ThreadBlockShape:kK, bot got group_size: " << args.group_size
                << ", ThreadBlockShape:kK: " << ThreadBlockShape::kK;
      exit(-1);
    }

    // Initialize the Params structure
    params_ = typename GemvKernel::Params{
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_scale.non_const_ref(),
      args.ref_zero_point.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      args.group_size,
      args.batch_count,
      args.batch_stride_A,
      args.batch_stride_B,
      args.batch_stride_scale,
      args.batch_stride_C,
      args.batch_stride_D
    };
  }

  /// Runs the kernel using initialized state.
  void run(cudaStream_t stream = nullptr) {

    dim3 grid = get_grid_size(params_);
    dim3 block = get_block_size(params_);

    cudaError_t result;

    int smem_size = int(sizeof(typename GemvKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemvKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        std::cout << "Gemv adapter cuda error: " << cudaGetErrorString(result)
                  << ", shared memory size: " << smem_size << "\n";
        exit(-1);
      }
    }

    cutlass::Kernel<GemvKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      std::cout << "Gemv adapter cuda  Launch error: " << cudaGetErrorString(result) << "\n";
      exit(-1);
    }
  }

  /// Runs the kernel using initialized state.
  void operator()(Arguments const &args, cudaStream_t stream = nullptr) {
    initialize(args);
    run(stream);
  }
};

}
}
}