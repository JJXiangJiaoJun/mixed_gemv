#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "gemv/weight_only_quant_op.h"

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
  WeightOnlyQuantOp QuantOp_,
  /// Used for partial specialization
  typename Enable = void
>
class DqMmaSingleStageGemv;

}
}
}

#include "gemv/threadblock/dq_mma_singlestage_finegrained.h"
#include "gemv/threadblock/dq_mma_singlestage_percol.h"