#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale_,
    /// Layout for the scale operand
    typename LayoutScale_,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Operation performed by GEMV
    typename Operator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Warp thread layout (concept: MatrixShape)
    typename WarpThreadArrangement,
    /// Operation performed by GEMM
    typename OperatorClass
>
struct DefaultDqMmaGemv;

}
}
}

#include "gemv/threadblock/default_dq_mma_simt.h"