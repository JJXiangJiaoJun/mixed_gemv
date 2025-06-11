/*! \file
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass
{

enum class WeightOnlyQuantOp
{
    UNDEFINED = 0,
    PER_COLUMN_SCALE_ONLY = 1,
    FINEGRAINED_SCALE_ONLY = 2,
    FINEGRAINED_SCALE_AND_ZEROS = 3
};

CUTLASS_HOST_DEVICE
constexpr bool isFinegrained(WeightOnlyQuantOp op)
{
    return op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS || op == WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
}

CUTLASS_HOST_DEVICE
constexpr bool hasZero(WeightOnlyQuantOp op)
{
    return op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
}

} // namespace cutlass