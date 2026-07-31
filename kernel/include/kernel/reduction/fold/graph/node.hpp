#pragma once

#include <kernel/reduction/fold/operation.hpp>

namespace rund::kernel {

struct FoldGraphEdge {
  u32 level = 0u;
  u32 left_slot = 0u;
  u32 right_slot = 0u;
  u32 output_slot = 0u;
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldPaddingLaw padding_law = FoldPaddingLaw::None;
  u64 padding_value = 0u;
  bool right_is_padding = false;
};

enum class FoldGraphNodeKind : u8 {
  WorkerLocalPartial = 0,
  GlobalOrderedSlot = 1,
  Reduction = 2,
};

struct FoldGraphNode {
  FoldGraphNodeKind kind = FoldGraphNodeKind::WorkerLocalPartial;
  u32 topological_index = 0u;
  u32 slot = 0u;
  u32 left_slot = 0u;
  u32 right_slot = 0u;
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldValueDomain value_domain = FoldValueDomain::HashDigest;
  FoldPaddingLaw padding_law = FoldPaddingLaw::None;
  FoldOverflowLaw overflow_law = FoldOverflowLaw::None;
  bool right_is_padding = false;
};

} // namespace rund::kernel
