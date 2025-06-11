#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "gemv/weight_only_quant_op.h"

namespace cutlass
{
namespace gemm
{
namespace warp
{

template<
  /// Matrix multiply operator
  typename MmaOperator,
  typename Element,
  WeightOnlyQuantOp QuantOp,
  int Count,
  typename HasZero = void
>
struct MmaOpDequantizer;

template<
  typename MmaOperator_,
  typename Element_,
  WeightOnlyQuantOp QuantOp,
  int Count
>
struct MmaOpDequantizer<
  MmaOperator_,
  Element_,
  QuantOp,
  Count,
  std::enable_if_t<hasZero(QuantOp)>
>{
public:

  using MmaOperator = MmaOperator_;
  using Element = Element_;

  static const int kCount = Count;
  static const WeightOnlyQuantOp kQuantOp = QuantOp;

  using Fragment = cutlass::Array<Element, kCount>;

  CUTLASS_DEVICE
  Fragment operator()(Fragment &operand_frag, Fragment &scale_frag, Fragment &zero_point_frag) {
    multiplies<Fragment> mul_op;
    plus<Fragment> plus_op;
    Fragment output;
    output = plus_op(mul_op(operand_frag, scale_frag), zero_point_frag);
    return output;
  }

};

template<
  typename MmaOperator_,
  typename Element_,
  WeightOnlyQuantOp QuantOp,
  int Count
>
struct MmaOpDequantizer<
  MmaOperator_,
  Element_,
  QuantOp,
  Count,
  std::enable_if_t<!hasZero(QuantOp)>
>{
public:

  using MmaOperator = MmaOperator_;
  using Element = Element_;

  static const int kCount = Count;
  static const WeightOnlyQuantOp kQuantOp = QuantOp;

  using Fragment = cutlass::Array<Element, kCount>;

  CUTLASS_DEVICE
  Fragment operator()(Fragment &operand_frag, Fragment &scale_frag, Fragment &zero_point_frag) {
    multiplies<Fragment> mul_op;
    Fragment output;
    output = mul_op(operand_frag, scale_frag);
    return output;
  }

};

}
}
}