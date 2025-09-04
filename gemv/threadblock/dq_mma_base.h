#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

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
  typename TransformAfterLDG_
>
class DqMmaBase{
public:
  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using IteratorScale = IteratorScale_; ///< Iterates over tiles of Scale operand in global memory
  using IteratorZeroPoint = IteratorZeroPoint_; ///< Iterates over tiles of Zero-point operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details
  using TransformAfterLDG = TransformAfterLDG_;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using FragmentScale = typename IteratorScale::Fragment;
  using FragmentZeroPoint = typename IteratorZeroPoint::Fragment;

  using TransformFragmentA = typename TransformAfterLDG::result_type;

  /// Fragment of accumulator tile
  using FragmentC = typename Operator::FragmentC;


  using WarpTileIteratorA = typename Operator::IteratorA;
  using WarpTileIteratorB = typename Operator::IteratorB;

  using WarpTileFragmentA = typename Operator::FragmentA;
  using WarpTileFragmentB = typename Operator::FragmentB;


  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Operator::Shape;

  /// FragmentB broadcast along M dimension
  static const int kBroadcastFactor = TransformFragmentA::kElements / FragmentB::kElements;

  template<
    typename T,
    int N,
    int Factor
  >
  struct RegisterBroadcaster {

    using Element = T;

    static const int kElements = N;
    static const int kFactor = Factor;

    using FragmentInput = cutlass::Array<T, kElements>;
    using FragmentBroadcast = cutlass::Array<T, kElements * kFactor>;

    CUTLASS_DEVICE
    FragmentBroadcast operator()(const FragmentInput& frag_input) {
      FragmentBroadcast frag_broadcast;
      FragmentInput * frag_broadcast_ptr = reinterpret_cast<FragmentInput *>(&frag_broadcast);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kFactor; ++i) {
        *(frag_broadcast_ptr + i) = frag_input;
      }
      return frag_broadcast;
    }

  };

  using OperandBBroadcaster = RegisterBroadcaster<typename FragmentB::Element, FragmentB::kElements, kBroadcastFactor>;
  using BroadcastedFragmentB  = typename OperandBBroadcaster::FragmentBroadcast;

  static_assert(std::is_same_v<FragmentB, typename OperandBBroadcaster::FragmentInput>, "");

public:
  CUTLASS_DEVICE
  DqMmaBase() {}

};

}
}
}