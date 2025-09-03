#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "arch/mma.h"
#include "layout/tile_interleaved_layout.h"
#include "gemv/interleaved_numeric_conversion.h"
#include "gemv/weight_only_quant_op.h"

#include "gemv/threadblock/dq_mma_singlestage.h"
#include "gemv/threadblock/default_dq_mma.h"
#include "gemv/threadblock/default_mma_core_simt.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename MmaCoreThreadMapA_, typename LayoutA_, typename Element_, typename Layout_, int Alignment>
struct DefaultScaleIteratorsSimt {
public:

  using MmaCoreThreadMapA = MmaCoreThreadMapA_;
  using LayoutA = LayoutA_;
  using Element = Element_;
  using Layout = Layout_;

  using MmaShape = typename MmaCoreThreadMapA::Shape;
  using MmaWarpArrangement = typename MmaCoreThreadMapA::Detail::WarpArrangement;
  using MmaWarpThreadArrangement = typename MmaCoreThreadMapA::Detail::WarpThreadArrangement;

  static constexpr int kRowsInterleaved = LayoutA::kRowsInterleaved;
  static constexpr int kColumnsPerTile = LayoutA::kColumnsPerTile;

  static const int kThreads = MmaCoreThreadMapA::kThreads;
  static const int kAlignment = Alignment;

  static_assert(kAlignment == kRowsInterleaved, "");

  using IteratorScaleThreadMap =
  cutlass::transform::threadblock::PitchLinearWarpStripminedStrideFirstBroadcastThreadMap<
     cutlass::PitchLinearShape<MmaShape::kStrided, 1>,  ///< (M, 1)
     kThreads,
     cutlass::PitchLinearShape<MmaWarpArrangement::kStrided, MmaWarpArrangement::kContiguous>,
     cutlass::PitchLinearShape<MmaWarpThreadArrangement::kStrided, MmaWarpThreadArrangement::kContiguous>,
     kAlignment
  >;

  using IteratorScale = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaShape::kStrided, 1>,
                                                                                Element, Layout, 1, IteratorScaleThreadMap, Alignment>;

  using IteratorZeroPoint = IteratorScale;

};


template<
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  /// Element type for the input scale
  typename ElementScale_,
  /// Layout for the scale operand
  typename LayoutScale_,
  /// Access granularity of Scales in unit of elements
  int AlignmentScale,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Operation performed by GEMM
  typename Operator_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement_
>
struct DefaultDqMmaGemv<
 ElementA_,
 LayoutA_,
 AlignmentA,
 ElementB_,
 cutlass::layout::ColumnMajor,
 AlignmentB,
 ElementScale_,
 LayoutScale_,
 AlignmentScale,
 ElementAccumulator_,
 ArchTag_,
 Operator_,
 ThreadblockShape_,
 WarpShape_,
 GemmShape<1, 1, 1>,
 WarpThreadArrangement_,
 arch::OpClassSimt
>{

public:


  using ElementA = ElementA_;
  using LayoutA  = LayoutA_;
  using ElementB = ElementB_;
  using ElementScale = ElementScale_;
  using LayoutScale = LayoutScale_;
  using ElementAccumulator = ElementAccumulator_;
  using ArchTag = ArchTag_;
  using Operator = Operator_;

  static_assert(platform::is_same<ElementA, uint8_t>::value || platform::is_same<ElementA, uint4b_t>::value,
      "Element A must be uint8 or uint4");

  static_assert(platform::is_same<ElementB, half_t>::value,
      "Element B must be half_t");

  using MmaCoreElementB = ElementB;
  using MmaCoreElementA = MmaCoreElementB;

  using ThreadBlockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using WarpThreadArrangement = WarpThreadArrangement_;

  using DeTagedOperator = typename cutlass::arch::DetagOperator<Operator>::Operator;

  static const int kAlignmentA = AlignmentA;
  static const int kAlignmentB = AlignmentB;
  static const int kAlignmentScale = AlignmentScale;

  static const WeightOnlyQuantOp QuantOp = cutlass::arch::DetagOperator<Operator>::QuantOp;

  static_assert(cutlass::layout::IsRowMajorTileInterleave<LayoutA>::value, "");

  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCoreGemv<ThreadBlockShape,
                                                              WarpShape,
                                                              InstructionShape,
                                                              WarpThreadArrangement,
                                                              MmaCoreElementA,
                                                              cutlass::layout::RowMajor,
                                                              MmaCoreElementB,
                                                              cutlass::layout::ColumnMajor,
                                                              ElementAccumulator,
                                                              cutlass::layout::RowMajor,
                                                              arch::OpClassSimt>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
                                                              ElementB,
                                                              cutlass::layout::ColumnMajor,
                                                              0,
                                                              typename MmaCore::IteratorThreadMapB,
                                                              kAlignmentB>;


  using IteratorB_LDG =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::IteratorThreadMapB_LDG::Shape::kStrided>,
                                                              ElementB,
                                                              cutlass::layout::ColumnMajor,
                                                              0,
                                                              typename MmaCore::IteratorThreadMapB_LDG,
                                                              kAlignmentB>;

private:

  static constexpr int kRowsInterleaved = LayoutA::kRowsInterleaved;
  static constexpr int kColumnsPerTile = LayoutA::kColumnsPerTile;

  static_assert(!(MmaCore::Shape::kM % kRowsInterleaved), "");
  static_assert(MmaCore::IteratorThreadMapA::Iterations::kContiguous == 1, "");
  static_assert(MmaCore::IteratorThreadMapB::Iterations::kContiguous == 1, "");
  static_assert(kColumnsPerTile == MmaCore::IteratorThreadMapB::kElementsPerAccess, "");

  // static_assert(MmaCore::WarpCount::kK == 1, "");

  using OriginalThreadMap = typename MmaCore::IteratorThreadMapA;
  using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;

  using GmemIteratorShape
        = MatrixShape<MmaCore::Shape::kM / kRowsInterleaved, MmaCore::Shape::kK * kRowsInterleaved>;

  using IteratorThreadMapA = cutlass::transform::threadblock::PitchLinearWarpStripminedThreadMap<
    cutlass::PitchLinearShape<GmemIteratorShape::kColumn, GmemIteratorShape::kRow>,
    OriginalThreadMap::kThreads,
    typename OriginalThreadMap::Detail::WarpArrangement,
    typename OriginalThreadMap::Detail::WarpThreadArrangement,
    MmaCore::kAccessSizeInBits / sizeof_bits<ElementA>::value
  >;

public:

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<GmemIteratorShape,
                                                              ElementA,
                                                              cutlass::layout::RowMajor,
                                                              1,
                                                              IteratorThreadMapA,
                                                              kAlignmentA>;


  using ScaleIterators = DefaultScaleIteratorsSimt<typename MmaCore::IteratorThreadMapA, LayoutA, ElementScale, LayoutScale, kAlignmentScale>;

  using IteratorScale = typename ScaleIterators::IteratorScale;

  using IteratorZeroPoint = typename ScaleIterators::IteratorZeroPoint;

  using TransformAfterLDG =
      cutlass::FastInterleavedAndBiasedNumericArrayConverter<MmaCoreElementA,
                                                             typename IteratorA::Element,
                                                             IteratorA::Fragment::kElements>;

  using ThreadBlockMma =
      cutlass::gemm::threadblock::DqMmaSingleStageGemv<typename MmaCore::Shape,
                                                       IteratorA,
                                                       IteratorB,
                                                       IteratorB_LDG,
                                                       IteratorScale,
                                                       IteratorZeroPoint,
                                                       ElementAccumulator,
                                                       cutlass::layout::RowMajor,
                                                       typename MmaCore::MmaPolicy,
                                                       TransformAfterLDG,
                                                       QuantOp>;
};

}
}
}