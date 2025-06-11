#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/activation.h"
#include "utils/vector_subbyte.h"

namespace host {
namespace reference {

template<
  typename ElementA,
  typename ElementB,
  typename ElementScale,
  typename ElementZero,
  typename ElementOutput,
  typename ElementAccumulate,
  typename ElementCompute = ElementAccumulate, // when Activation type is half, it must be cutlass::half_t
  typename Activation = cutlass::epilogue::thread::Identity<ElementCompute>,
  bool isSubbytesA = (cutlass::sizeof_bits<ElementA>::value < 8)
>
struct MixedGemv;

template<
 typename ElementA_,
 typename ElementB_,
 typename ElementScale_,
 typename ElementOutput_,
 typename ElementAccumulate_,
 typename ElementCompute_,
 typename Activation_
>
struct MixedGemv<
 ElementA_,
 ElementB_,
 ElementScale_,
 int8_t,
 ElementOutput_,
 ElementAccumulate_,
 ElementCompute_,
 Activation_,
 true
>{
public:
 using ElementA = ElementA_;
 using ElementB = ElementB_;
 using ElementScale = ElementScale_;
 using ElementZero = int8_t;
 using ElementOutput = ElementOutput_;
 using ElementAccumulate = ElementAccumulate_;
 using ElementCompute = ElementCompute_;
 using Activation = Activation_;

 static const bool kIsSubbyteA = true;
 Activation act;

public:

  MixedGemv() {}

 void operator()(int32_t batch_size,
                 int32_t M,
                 int32_t N,
                 int32_t K,
                 int32_t group_size,
                 const host::utils::vector_subbyte<ElementA>& A,
                 const ElementB* B,
                 const ElementScale* scale,
                 const ElementZero* zero_point,
                 const ElementOutput* bias,
                 ElementOutput* output) {

   if (N != 1) {
    std::cout << "N must be 1 in gemv, but got " << N << std::endl;
    exit(-1);
   }

   if (batch_size != 1) {
    std::cout << "Batch size must be 1, but got " << batch_size << std::endl;
    exit(-1);
   }

    for (int m_i = 0; m_i < M; ++m_i) {

      ElementAccumulate acc = ElementAccumulate(0);

      for (int k_i = 0; k_i < K; ++k_i) {

        int group_idx = k_i / group_size;

          ///< Dequantize for weight
          ///< - scale * zero
          ElementScale scale_x_zero = -(scale[group_idx * M +  m_i]
                                        * static_cast<ElementScale>(zero_point[group_idx * M + m_i]));
          ///< q * scale - scale * zero
          ElementAccumulate dq_A = static_cast<ElementAccumulate>(static_cast<ElementScale>(
                                   int(A[m_i * K + k_i].get())) * scale[group_idx * M + m_i] + scale_x_zero);

          acc += static_cast<ElementAccumulate>(B[0 * K + k_i]) * dq_A;

      }

      ElementCompute bias_val = (bias != nullptr) ? static_cast<ElementCompute>(bias[m_i]) : ElementCompute(0);
      output[m_i * 1 + 0] = act(static_cast<ElementCompute>(acc) + bias_val);
    }
 }
};




} /// reference
} ///  host