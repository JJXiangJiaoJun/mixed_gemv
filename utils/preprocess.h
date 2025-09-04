#pragma once

#include <limits>
#include <iostream>
#include "stdint.h"

#include "cutlass/integer_subbyte.h"

#include "utils/cuda_utils.h"
#include "utils/vector_subbyte.h"
#include "utils/type_base.h"

namespace host {
namespace utils {

template<
 typename ElementSubbyte,
 typename ElementUnpacked,
 int SubbyteBits = cutlass::sizeof_bits<ElementSubbyte>::value,
 bool Signed = true
>
struct SubBytePacker;

template<
 typename ElementSubbyte_
>
struct SubBytePacker<
 ElementSubbyte_,
 int8_t,
 4,
 true
>{
public:

  using ElementSubbyte = ElementSubbyte_;  ///< must be int4b_t
  using ElementUnpacked = int8_t;
  using ElementPackedStorage = ElementUnpacked;
  using size_type = size_t;

  ///< unpack
  static void unpack(ElementUnpacked* unpacked_output,
                     const ElementPackedStorage* packed_input,
                     size_type packed_num_elements) {
    for (size_type packed_idx = 0; packed_idx < packed_num_elements; ++packed_idx) {

      ElementUnpacked packed_data = packed_input[packed_idx];
      ElementUnpacked elt_0 = (ElementUnpacked(packed_data << 4) >> 4);  // The double shift here is to ensure sign extension
      ElementUnpacked elt_1 = packed_data >> 4;

      unpacked_output[2 * packed_idx + 0] = elt_0;
      unpacked_output[2 * packed_idx + 1] = elt_1;
    }
  }

  static void pack(ElementPackedStorage* packed_output,
                   const ElementUnpacked* unpacked_input,
                   size_type unpacked_num_elements) {

    ElementPackedStorage kLowest = std::numeric_limits<ElementSubbyte>::lowest();
    ElementPackedStorage kMax = std::numeric_limits<ElementSubbyte>::max();

    size_type packed_num_elements = unpacked_num_elements / 2;

    for (size_type packed_idx = 0; packed_idx < packed_num_elements; ++packed_idx) {
      ElementPackedStorage packed_int4s = 0;
      ElementPackedStorage elt_0 = unpacked_input[2 * packed_idx + 0];
      ElementPackedStorage elt_1 = unpacked_input[2 * packed_idx + 1];

      if (elt_0 < kLowest && elt_0 > kMax) {
        std::cout << "elt0: " << int(elt_0) << " in unpacked tensor not in int4 range" << std::endl;
        exit(-1);
      }

      if (elt_1 < kLowest && elt_1 > kMax) {
        std::cout << "elt_1: " << int(elt_1) << " in unpacked tensor not in int4 range" << std::endl;
        exit(-1);
      }

      packed_int4s |= ((elt_0 & 0x0F));
      packed_int4s |= int8_t(elt_1 << 4);

      packed_output[packed_idx] = packed_int4s;
    }
  }
};


template <typename ElementScale_, typename ElementZeroPoint_>
struct ZeroPointPreprocessor {
 public:
  using ElementScale = ElementScale_;
  using ElementZeroPoint = ElementZeroPoint_;

  static void neg_zero_point_multiplies_scale(ElementScale* neg_zeros_x_scales,
                                              const ElementScale* scales,
                                              const ElementZeroPoint* zeros,
                                              int n) {
    for (int i = 0; i < n; ++i) {
      neg_zeros_x_scales[i] = static_cast<ElementScale>(-zeros[i]) * scales[i];
    }
  }
};


constexpr int inline get_weight_quant_bits(QuantType quant_type) {
  switch (quant_type) {
    case QuantType::W8_A16: return weight_quant_bits<QuantType::W8_A16>::value;
    case QuantType::W4_A16: return weight_quant_bits<QuantType::W4_A16>::value;
    case QuantType::W4_AFP8: return weight_quant_bits<QuantType::W4_AFP8>::value;
    default: std::cout << "get_weight_quant_bits: invalid QuantType " << int(quant_type); return -1;
  }
}

////< reference from trtllm cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h

struct SubbyteWeightProcessor {
public:

  // We need to use this transpose to correctly handle packed int4 and int8 data
  // The reason this code is relatively complex is that the "trivial" loops took a substantial
  // amount of time to transpose leading to long preprocessing times. This seemed to be a big
  // issue for relatively large models.
  template <QuantType quant_type>
  static void subbyte_transpose_impl(
      int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor, std::vector<size_t> const& shape)
  {
      constexpr int bits_per_elt = get_weight_quant_bits(quant_type);

      if (shape.size() != 2 && shape.size() != 3) {
        std::cout << "Shape must be 2-D or 3-D, but got " << shape.size();
        exit(-1);
      }

      const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
      const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
      const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

      const size_t col_bytes = num_cols * bits_per_elt / 8;
      const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
      // const size_t num_bytes = size_t(num_experts) * num_rows * col_bytes;

      uint8_t const* input_byte_ptr = reinterpret_cast<uint8_t const*>(quantized_tensor);
      uint8_t* output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

      static constexpr int ELTS_PER_BYTE = 8 / bits_per_elt;

      static constexpr int M_TILE_L1 = 64;
      static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
      uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

      static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

      // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
      // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
      // allows GCC to emit vector instructions.

      if (col_bytes_trans % VECTOR_WIDTH || col_bytes % VECTOR_WIDTH) {
        std::cout << "Number of bytes for rows and cols must be a multiple of " << VECTOR_WIDTH
              << ", However, num_rows_bytes = " << col_bytes_trans
              << ", and num_col_bytes = " << col_bytes;
        exit(-1);
      }


      // int const num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
      // int const num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

      for (size_t expert = 0; expert < num_experts; ++expert)
      {
          const size_t matrix_offset = expert * num_rows * col_bytes;
          for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1)
          {
              for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes; col_tile_start_byte += N_TILE_L1)
              {

                  int const row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
                  int const col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

                  for (int ii = 0; ii < M_TILE_L1; ++ii)
                  {
                      int const row = row_tile_start + ii;

                      for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                      {
                          int const col = col_tile_start_byte + jj;

                          const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

                          if (row < row_limit && col < col_limit)
                          {
                              for (int v = 0; v < VECTOR_WIDTH; ++v)
                              {
                                  cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
                              }
                          }
                      }
                  }

                  if constexpr (bits_per_elt == 8)
                  {
                      for (int ii = 0; ii < M_TILE_L1; ++ii)
                      {
                          for (int jj = ii + 1; jj < N_TILE_L1; ++jj)
                          {
                              std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
                          }
                      }
                  }
                  else if constexpr (bits_per_elt == 4)
                  {

                      for (int ii = 0; ii < M_TILE_L1; ++ii)
                      {
                          // Using M_TILE_L1 here is deliberate since we assume that the cache tile
                          // is square in the number of elements (not necessarily the number of bytes).
                          for (int jj = ii + 1; jj < M_TILE_L1; ++jj)
                          {
                              int const ii_byte = ii / ELTS_PER_BYTE;
                              int const ii_bit_offset = ii % ELTS_PER_BYTE;

                              int const jj_byte = jj / ELTS_PER_BYTE;
                              int const jj_bit_offset = jj % ELTS_PER_BYTE;

                              uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
                              uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

                              cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
                              cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

                              cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
                              cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
                          }
                      }
                  }
                  else
                  {
                      std::cout << "Unsupported quantization type, bits per element: " << bits_per_elt;
                      exit(-1);
                  }

                  const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
                  const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

                  int const row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
                  int const col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

                  for (int ii = 0; ii < M_TILE_L1; ++ii)
                  {
                      int const row = row_tile_start_trans + ii;
                      for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                      {
                          int const col = col_tile_start_byte_trans + jj;

                          const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

                          if (row < row_limit_trans && col < col_limit_trans)
                          {
                              for (int v = 0; v < VECTOR_WIDTH; ++v)
                              {
                                  output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
                              }
                          }
                      }
                  }
              }
          }
      }
  }

  static void subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
                                std::vector<size_t> const& shape, QuantType quant_type)
  {
      if (quant_type == QuantType::W8_A16)
      {
          subbyte_transpose_impl<QuantType::W8_A16>(transposed_quantized_tensor, quantized_tensor, shape);
      }
      else if (quant_type == QuantType::W4_A16)
      {
          subbyte_transpose_impl<QuantType::W4_A16>(transposed_quantized_tensor, quantized_tensor, shape);
      }
      else if (quant_type == QuantType::W4_AFP8)
      {
          subbyte_transpose_impl<QuantType::W4_AFP8>(transposed_quantized_tensor, quantized_tensor, shape);
      }
      else
      {
          std::cout << "Invalid Quant Type: " << int(quant_type);
          exit(-1);
      }
  }

  template <QuantType quant_type>
  static void interleave_row_major_tensor_impl(int8_t* interleaved_quantized_tensor, int8_t const* quantized_tensor, std::vector<size_t> const& shape) {
    constexpr int kQuantizedBits = host::weight_quant_bits<quant_type>::value;
    constexpr int kActBits = host::activation_quant_bits<quant_type>::value;

    static_assert((kActBits % kQuantizedBits) == 0, "Activation bits must be divided by kQuantized bits");

    constexpr int kInterleave = kActBits / kQuantizedBits;
    constexpr int kGlobalLoadPerAccessBits = 16 * 8;
    constexpr int kActAlignment = kGlobalLoadPerAccessBits / kActBits; // 16 * 8 // 16 = 8
    constexpr int kQuantizedLoadBits = kQuantizedBits * kActAlignment; // 4 * 8 = 32;
    static_assert(kQuantizedLoadBits * kInterleave == kGlobalLoadPerAccessBits, "");

    constexpr int kElementsPerAccess = kQuantizedLoadBits / kQuantizedBits;
    constexpr int kBitsPerAccess = kElementsPerAccess * kQuantizedBits;

    using ElementAccess = typename host::AccessTypeDeduce<kBitsPerAccess>::type;

    if (shape.size() != 2 && shape.size() != 3) {
      std::cout << "interleave_row_major_tensor_impl Shape must be 2-D or 3-D, but got " << shape.size() << std::endl;
      exit(-1);
    }

    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_columns = shape.size() == 2 ? shape[1] : shape[2];

    if (num_rows % kInterleave) {
      std::cout << "Shape Row must be divided by interleave: " << kInterleave << ", but got " << num_rows  << std::endl;
      exit(-1);
    }

    if (num_columns % kElementsPerAccess) {
      std::cout << "Shape columns must be divided by kElementsPerAccess: " << kElementsPerAccess << ", but got " << num_columns  << std::endl;
      exit(-1);
    }

    int num_columns_in_access = num_columns / kElementsPerAccess;

    ElementAccess const * input_ptr = reinterpret_cast<ElementAccess const *>(quantized_tensor);
    ElementAccess * output_ptr = reinterpret_cast<ElementAccess *>(interleaved_quantized_tensor);

    for (size_t expert = 0; expert < num_experts; ++expert) {

      const size_t matrix_offset = expert * num_rows * num_columns_in_access;

      for (int64_t r = 0; size_t(r) < num_rows; ++r) {
        for (int64_t c = 0; c < int64_t(num_columns_in_access); ++c) {
          int64_t input_offset  = matrix_offset + r * num_columns_in_access + c;
          int64_t row_major = r / kInterleave;
          int64_t row_minor = r % kInterleave;
          int64_t output_offset = matrix_offset + row_major * (num_columns_in_access * kInterleave) + c * kInterleave + row_minor;
          *(output_ptr + output_offset) = *(input_ptr + input_offset);
        }
      }
    }
  }

  static void interleave_row_major_tensor(int8_t* interleaved_quantized_tensor,
                                          int8_t const* quantized_tensor,
                                          std::vector<size_t> const& shape, QuantType quant_type) {
    if (quant_type == QuantType::W4_A16)
    {
        interleave_row_major_tensor_impl<QuantType::W4_A16>(interleaved_quantized_tensor, quantized_tensor, shape);
    }
    else
    {
        std::cout << "interleave_row_major_tensor invalid Quant Type : " << int(quant_type) << std::endl;
        exit(-1);
    }
  }


  static void add_bias_and_interleave_int8s_inplace(int8_t* int8_tensor, const size_t num_elts)
  {
      for (int ii = 0; size_t(ii) < num_elts; ++ii)
      {
          int8_tensor[ii] = int8_t(int(int8_tensor[ii]) + 128);
      }

      // Step 2 will transform the layout of a 32-bit register in CUDA in order to match the int4 layout. This has no
      // performance benefit and is purely so that int4 and int8 have the same layout.
      // Pictorially, this does the following:
      // bit 32                                                      0
      //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
      //
      // And it will rearrange the output 32 bit register to be the following:
      // bit 32                                                      0
      //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

      if (num_elts % 4) {
        std::cout << "Dimensions of int8 tensor must be a multiple of 4 for register relayout, but got: " << num_elts;
        exit(-1);
      }

      for (size_t base = 0; base < num_elts; base += 4)
      {
          std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
      }
  }

  static void add_bias_and_interleave_int4s_inplace(int8_t* packed_int4_tensor, const size_t num_elts)
  {
      int const num_bytes = num_elts / 2;

      // Step 1 will be to transform all the int4s to unsigned in order to make the dequantize take as little
      // instructions as possible in the CUDA code.
      for (size_t ii = 0; ii < size_t(num_bytes); ++ii)
      {
          int8_t transformed_packed_int4s = 0;
          int8_t transformed_first_elt
              = (int8_t(packed_int4_tensor[ii] << 4) >> 4) + 8; // The double shift here is to ensure sign extension
          int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4) + 8;

          if (transformed_first_elt < 0 || transformed_first_elt > 15) {
            std::cout << "Illegal result for int4 transform (first elt): " << int(transformed_first_elt);
            exit(-1);
          }

          if (transformed_second_elt < 0 || transformed_second_elt > 15) {
            std::cout << "Illegal result for int4 transform (second elt): " << int(transformed_second_elt);
            exit(-1);
          }

          // We don't need to mask in these ops since everything should be in the range 0-15
          transformed_packed_int4s |= transformed_first_elt;
          transformed_packed_int4s |= (transformed_second_elt << 4);
          packed_int4_tensor[ii] = transformed_packed_int4s;
      }

      // Step 2 will transform the layout of a 32-bit register in CUDA in order to minimize the number of shift & logical
      // instructions That are needed to extract the int4s in the GEMM main loop. Pictorially, the loop below will do the
      // following: Take as input a 32 bit register with layout: bit 32 0
      //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 4 bits)
      //
      // And it will rearrange the output 32 bit register to be the following:
      // bit 32                                                      0
      //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)

      if (num_elts % 8) {
        std::cout << "Dimensions of int4 tensor must be a multiple of 8 for register relayout, but got: " << num_elts;
        exit(-1);
      }

      const size_t num_registers = num_bytes / 4;

      uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int4_tensor);
      for (size_t ii = 0; ii < num_registers; ++ii)
      {
          const uint32_t current_register = register_ptr[ii];
          uint32_t transformed_register = 0;

          for (int dest_idx = 0; dest_idx < 8; ++dest_idx)
          {
              int const src_idx = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
              int const src_shift = 4 * src_idx;
              int const dest_shift = 4 * dest_idx;

              const uint32_t src_bits = (current_register >> src_shift) & 0xF;
              transformed_register |= (src_bits << dest_shift);
          }
          register_ptr[ii] = transformed_register;
      }
  }

  static void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type)
  {
      if (quant_type == QuantType::W8_A16)
      {
          add_bias_and_interleave_int8s_inplace(tensor, num_elts);
      }
      else if (quant_type == QuantType::W4_A16 || quant_type == QuantType::W4_AFP8)
      {
          // W4_AFP8 uses the same preprocessor as W4_A16 because the FP8 data must
          // be converted to FP16 before the scales can be applied using CUDA cores.
          // As a result, we still want permute the data so that it is well aligned
          // for conversion to FP16.
          add_bias_and_interleave_int4s_inplace(tensor, num_elts);
      }
      else
      {
          std::cout << "Invalid QuantType: " << int(quant_type) << ", for interleaving.";
      }
  }

};


struct MixedGemvWeightPreprocessor {
public:
  static void run(int8_t* preprocessed_quantized_weight, int8_t const* quantized_weight,
                  std::vector<size_t> const& shape, LayoutType weight_layout,
                  QuantType quant_type, bool force_interleave) {

    if (shape.size() != 2 && shape.size() != 3) {
      std::cout << "Mixed-Gemv weight Shape must be 2-D or 3-D" << shape.size() << std::endl;
      exit(-1);
    }

    size_t num_elts = 1;
    for (auto const& dim : shape)
    {
        num_elts *= dim;
    }

    const size_t num_bytes = num_elts * get_weight_quant_bits(quant_type) / 8;

    std::vector<int8_t> src_buf(num_bytes);
    std::vector<int8_t> dst_buf(num_bytes);
    std::copy(quantized_weight, quantized_weight + num_bytes, src_buf.begin());

    std::vector<size_t> weight_shape = shape;

    // Works on row major data, so issue this permutation first
    // [K, M] RowMajor to [K, M] ColMajor
    if (weight_layout == LayoutType::kRowMajor)
    {
        SubbyteWeightProcessor::subbyte_transpose(dst_buf.data(), src_buf.data(), weight_shape, quant_type);
        src_buf.swap(dst_buf);
    }

    ///< transposed [K, M] --> [M, K]
    std::swap(weight_shape[weight_shape.size() - 1], weight_shape[weight_shape.size() - 2]);

    ///< interleaved (M, K) row-major
    SubbyteWeightProcessor::interleave_row_major_tensor(dst_buf.data(), src_buf.data(), weight_shape, quant_type);
    src_buf.swap(dst_buf);

    SubbyteWeightProcessor::add_bias_and_interleave_quantized_tensor_inplace(src_buf.data(), num_elts, quant_type);
    std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
  }

};

} // end of namespace utils
} // end of namespace host