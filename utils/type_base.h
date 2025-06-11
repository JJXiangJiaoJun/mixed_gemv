#pragma once

#include <stdint.h>

namespace host {

enum class QuantType {
  W8_A16 = 0,
  W4_A16 = 1,
  W4_AFP8 = 2
};

enum class LayoutType {
  kRowMajor = 0,
  kColumnMajor = 1
};


template<QuantType Type>
struct weight_quant_bits;

template<QuantType Type>
struct activation_quant_bits;

template<>
struct  weight_quant_bits<QuantType::W8_A16> {
  static constexpr int value = 8;
};

template<>
struct  weight_quant_bits<QuantType::W4_A16> {
  static constexpr int value = 4;
};

template<>
struct  weight_quant_bits<QuantType::W4_AFP8> {
  static constexpr int value = 4;
};

template<>
struct  activation_quant_bits<QuantType::W8_A16> {
  static constexpr int value = 16;
};

template<>
struct  activation_quant_bits<QuantType::W4_A16> {
  static constexpr int value = 16;
};

template<>
struct  activation_quant_bits<QuantType::W4_AFP8> {
  static constexpr int value = 8;
};

template<int Bits>
struct AccessTypeDeduce;

template<>
struct AccessTypeDeduce<32> {
  using type = int32_t;
};

}