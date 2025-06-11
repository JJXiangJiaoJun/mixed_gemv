#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/array.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "arch/mma.h"
#include "gemv/weight_only_quant_op.h"
#include "layout/tile_interleaved_layout.h"

#include "utils/initializer.h"
#include "utils/preprocess.h"
#include "utils/vector_subbyte.h"
#include "host/mixed_gemv.h"

#include "gemv/device/gemv_adaptor.h"

// #define PRINT_DEBUG
// #define HOST_CHECK

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::half_t;
using DeviceElementA = cutlass::uint4b_t;
using ElementScale = cutlass::half_t;
using ElementZero = int8_t;
using ElementScaleZero = ElementScale;

static const int kGemmN = 1;
static const int kStages = 1;
static const int kAlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
static const int kAlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;

static const int kInterleave = kAlignmentA / kAlignmentB;

static_assert(kInterleave == 4, "");

static const int kAlignmentScale = kInterleave;
static const int kAlignmentC = kInterleave;

using LayoutA  = cutlass::layout::RowMajorTileInterleave<kAlignmentB, kAlignmentA / kAlignmentB>;
using LayoutB  = cutlass::layout::ColumnMajor;
using LayoutScale = cutlass::layout::ColumnMajor;

using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;
using ElementCompute = ElementAccumulator;

using OperatorClass = cutlass::arch::OpClassSimt;
using ArchTag = cutlass::arch::Sm50;

using ElementUnpacked = int8_t;
using ElementStorage = ElementUnpacked;
using ElementAPackType = host::utils::vector_subbyte<ElementA>;

using ThreadBlockShape = cutlass::gemm::GemmShape<64, 1, 64>;
using WarpShape = cutlass::gemm::GemmShape<16, 1, 64>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using WarpThreadArrangement = cutlass::MatrixShape<4, 8>;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,            // <- data type of output matrix
    kAlignmentC,         // <- this is the number of elements per
                         // vectorized memory access. For half
                         // precision, it's 8 elements. This becomes
                         // the vector width of math instructions in
                         // epilogue too
    ElementAccumulator,  // <- data type of accumulator
    ElementCompute,  // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;  // <- alpha x C + bias


static const cutlass::WeightOnlyQuantOp QuantOp = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
// static const cutlass::WeightOnlyQuantOp QuantOp = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
// static const cutlass::WeightOnlyQuantOp QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

using TaggedOperator = cutlass::arch::TagOperator<cutlass::arch::OpMultiplyAdd, QuantOp>::TaggedOperator;


using DeviceKernel = cutlass::gemm::device::GemvAdaptor<DeviceElementA,
                                                        LayoutA,
                                                        ElementB,
                                                        LayoutB,
                                                        ElementScale,
                                                        LayoutScale,
                                                        ElementC,
                                                        LayoutC,
                                                        ElementAccumulator,
                                                        OperatorClass,
                                                        ArchTag,
                                                        TaggedOperator,
                                                        ThreadBlockShape,
                                                        WarpShape,
                                                        InstructionShape,
                                                        WarpThreadArrangement,
                                                        EpilogueOutputOp,
                                                        kStages,
                                                        kAlignmentA,
                                                        kAlignmentB,
                                                        kAlignmentScale>;

using HostKernel = host::reference::MixedGemv<
 ElementA,
 ElementB,
 ElementScale,
 ElementZero,
 ElementC,
 ElementAccumulator,
 ElementCompute
>;


void device_gemv(const DeviceElementA *ptr_A,
                 const ElementB *ptr_B,
                 const ElementScaleZero *ptr_scale,
                 const ElementScaleZero *ptr_zero_point,
                 const ElementC *ptr_C,
                 ElementC *ptr_D,
                 int B,
                 int M,
                 int N,
                 int K,
                 int group_size) {

  int lda = K * LayoutA::kRowsInterleaved;
  int ldb = 0;
  int ldc = M;
  // int ldc = 15461;
  int ldd = M;

  if (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
    group_size = K;

  if (QuantOp != cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
    ptr_zero_point = nullptr;
  }

  typename DeviceKernel::Arguments args{
    cutlass::make_Coord(M, N, K),
    {reinterpret_cast<DeviceElementA*>(const_cast<DeviceElementA*>(ptr_A)), lda},
    cutlass::make_TensorRef(ptr_B, LayoutB(ldb)),
    cutlass::make_TensorRef(ptr_scale, LayoutScale(M)),
    cutlass::make_TensorRef(ptr_zero_point, LayoutScale(M)),
    // cutlass::make_TensorRef(ptr_C, LayoutC(ldc)),
    {nullptr, ldc},
    cutlass::make_TensorRef(ptr_D, LayoutC(ldd)),
    {},
    group_size
  };

  DeviceKernel op;
  op.initialize(args);
  op.run();
}

void host_gemv(const ElementAPackType &A,
               const ElementB *ptr_B,
               const ElementScale *ptr_scale,
               const ElementZero *ptr_zero_point,
               const ElementC *ptr_bias,
               ElementC *ptr_D,
               int B,
               int M,
               int N,
               int K,
               int group_size) {

  HostKernel host_op;
  host_op(B, M, N, K, group_size, A, ptr_B, ptr_scale, ptr_zero_point, ptr_bias, ptr_D);
}


int main() {

  int B = 1;
  int M = 151936, K = 2048;
  // int M = 2048, K = 11008;
  // int M = 11008, K = 2048;
  // int M = 2048, K = 2048;

  int group_size = 128;

  if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY) {
    group_size = K;
  }

  int total_group = (K + group_size - 1) / group_size;

  ///< Host A (m, k)
  ElementStorage *h_A_storage = new ElementStorage[M * K];
  reference::random_initializer<ElementStorage>::init(
      h_A_storage,
      M * K,
      cutlass::platform::numeric_limits<ElementA>::lowest(),
      cutlass::platform::numeric_limits<ElementA>::max());

  host::utils::vector_subbyte<ElementA> h_A(M * K, 0);

  host::utils::SubBytePacker<ElementA, ElementUnpacked>::pack((ElementUnpacked *)h_A.data(), h_A_storage, M * K);

  for (int i = 0; i < M * K; i++) {
    int32_t reference = static_cast<int32_t>(h_A_storage[i]);
    int32_t packed    = int32_t(h_A[i].get());
    if (reference != packed) {
      std::cout << "host pack diff: " << i << ", reference: " << reference << ", packed: " << packed << std::endl;
    }
  }

  ////< Host B (1, K) column-major
  ElementB *h_B = new ElementB[B * kGemmN * K];
  reference::random_initializer<ElementB>::init(h_B, B * kGemmN * K);

  ///< Host scale (total_group, M)
  ElementScale *h_scale = new ElementScale[total_group * M];
  reference::random_initializer<ElementScale>::init(h_scale, total_group * M, 0.1, 0.3);

  ///< Host Zero-point (total_group, M)
  ElementZero *h_zero_point = new ElementZero[total_group * M];

  if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
    reference::random_initializer<ElementZero>::init(h_zero_point, total_group * M, -8, 7);
  } else {
    reference::random_initializer<ElementZero>::init(h_zero_point, total_group * M, 0);
  }

  ///< zero-point preprocess
  ElementScaleZero *h_neg_zeros_x_scales = new ElementScaleZero[total_group * M];
  host::utils::ZeroPointPreprocessor<ElementScaleZero, ElementZero>::neg_zero_point_multiplies_scale(
      h_neg_zeros_x_scales, h_scale, h_zero_point, total_group * M);

  // ElementC *h_bias = new ElementC[M];
  ///< Host Output
  ElementC *h_D = new ElementC[B * M * kGemmN];

  ///< mixed_gemv weight preprocess
  host::utils::vector_subbyte<ElementA> h_A_processed(M * K, 0);
  host::utils::MixedGemvWeightPreprocessor::run(
    reinterpret_cast<int8_t *>(h_A_processed.raw_data()),
    reinterpret_cast<int8_t *>(h_A.raw_data()),
    {size_t(K), size_t(M)},
    host::LayoutType::kColumnMajor,
    host::QuantType::W4_A16,
    false
  );

  ElementC *result_D = new ElementC[B * M * kGemmN];

  DeviceElementA *d_A;
  ElementB *d_B;
  ElementScaleZero *d_scale, *d_zero_point;
  // ElementC *d_bias;
  ElementC *d_D;

  cudaMalloc(&d_A, M * K / ( 8 / cutlass::sizeof_bits<DeviceElementA>::value));
  cudaMalloc(&d_B, B * kGemmN * K * sizeof(ElementB));
  cudaMalloc(&d_scale, total_group * M * sizeof(ElementScaleZero));
  cudaMalloc(&d_zero_point, total_group * M * sizeof(ElementScaleZero));
  // cudaMalloc(&d_bias, M * sizeof(ElementC));
  cudaMalloc(&d_D, B * M * kGemmN * sizeof(ElementC));

  cudaMemcpy(d_A, h_A_processed.raw_data(), M * K / (8 / cutlass::sizeof_bits<DeviceElementA>::value), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, B * kGemmN * K * sizeof(ElementB), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, h_scale, total_group * M * sizeof(ElementScale), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_bias, h_bias, M * sizeof(ElementC), cudaMemcpyHostToDevice);
  cudaMemcpy(d_zero_point, h_neg_zeros_x_scales, total_group * M * sizeof(ElementScaleZero), cudaMemcpyHostToDevice);


  for (int i = 0; i < 10; i++)
  device_gemv(d_A, d_B, d_scale, d_zero_point, nullptr, d_D, B, M, kGemmN, K, group_size);

  cudaMemcpy(result_D, d_D, B * M * kGemmN * sizeof(ElementC), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaError_t result = cudaGetLastError();

  if (result != cudaSuccess) {
    std::cout << "Execution error: " << cudaGetErrorString(result) << std::endl;
    exit(-1);
  }

#ifdef HOST_CHECK

  host_gemv(h_A, h_B, h_scale, h_zero_point, nullptr, h_D, B, M, kGemmN, K, group_size);

  for (int m = 0; m < M; m++) {
    float abs_err = fabs(float(h_D[m]) - float(result_D[m]));
    if (abs_err > 1e-3) {
      std::cout <<"m: " << m << " cpu: " << float(h_D[m]) << "\tgpu: " << float(result_D[m]) << "\tdiff: " << abs_err << std::endl;
    }
  }

#endif



  // delete[] h_A;
  delete[] h_A_storage;
  delete[] h_B;

  // delete[] h_bias;
  delete[] h_scale;
  delete[] result_D;
  cudaFree(d_A);
  cudaFree(d_B);
  // cudaFree(d_bias);
  cudaFree(d_scale);
  cudaFree(d_D);

  return 0;
}