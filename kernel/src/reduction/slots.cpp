#include <kernel/reduction/fold/slots.hpp>

#include <algorithm>

namespace rund::kernel {

void FixedBinaryTreeHashBuilder::Push(u64 value) {
  values_.push_back(value);
}

u64 FixedBinaryTreeHashBuilder::Finalize() const {
  return FoldHashFixedBinaryTree(std::span<const u64>(values_.data(), values_.size()));
}

void FixedBinaryTreeHashBuilder::Reset() {
  values_.clear();
}

void ResetFoldSlots(FoldSlots& slots) {
  slots.values.clear();
}

bool ReserveFoldSlots(FoldSlots& slots, const u32 slot_capacity) {
  try {
    slots.values.reserve(slot_capacity);
  } catch (...) {
    return false;
  }
  return true;
}

void EnsureFoldSlots(FoldSlots& slots, const u32 slot_count) {
  if (slots.values.size() != static_cast<std::size_t>(slot_count)) {
    slots.values.resize(slot_count);
  }
  std::fill(slots.values.begin(), slots.values.end(), 0u);
}

bool StoreFoldSlot(FoldSlots& slots, const u32 slot, const u64 value) {
  if (slot >= slots.values.size()) {
    return false;
  }
  slots.values[slot] = value;
  return true;
}

std::span<u64> MutableFoldSlots(FoldSlots& slots) {
  return std::span<u64>(slots.values.data(), slots.values.size());
}

std::span<const u64> ViewFoldSlots(const FoldSlots& slots) {
  return std::span<const u64>(slots.values.data(), slots.values.size());
}

u64 FoldSlotsXor(const std::span<const u64> values) {
  return FoldOrderedSlots(values, FoldOperation::Xor).value;
}

u64 FoldSlotsMax(const std::span<const u64> values) {
  return FoldOrderedSlots(values, FoldOperation::Max).value;
}

u64 FoldSlotsMin(const std::span<const u64> values) {
  return FoldOrderedSlots(values, FoldOperation::Min).value;
}

u64 FoldSlotsSaturatingAdd(const std::span<const u64> values) {
  return FoldOrderedSlots(values, FoldOperation::SaturatingAdd).value;
}

u64 FoldHashFixedBinaryTree(const std::span<const u64> ordered_values) {
  return FoldOrderedSlots(ordered_values, FoldOperation::FixedBinaryTreeHash).value;
}

} // namespace rund::kernel
