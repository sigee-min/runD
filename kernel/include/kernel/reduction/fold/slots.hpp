#pragma once

#include <kernel/reduction/fold/result.hpp>
#include <kernel/reduction/fold/strict.hpp>

#include <span>
#include <vector>

namespace rund::kernel {

struct FoldSlots {
  std::vector<u64> values{};
};

class FixedBinaryTreeHashBuilder final {
public:
  void Push(u64 value);
  u64 Finalize() const;
  void Reset();

private:
  std::vector<u64> values_{};
};

void ResetFoldSlots(FoldSlots &slots);
bool ReserveFoldSlots(FoldSlots &slots, u32 slot_capacity);
void EnsureFoldSlots(FoldSlots &slots, u32 slot_count);
bool StoreFoldSlot(FoldSlots &slots, u32 slot, u64 value);
std::span<u64> MutableFoldSlots(FoldSlots &slots);
std::span<const u64> ViewFoldSlots(const FoldSlots &slots);
u64 FoldSlotsXor(std::span<const u64> values);
u64 FoldSlotsMax(std::span<const u64> values);
u64 FoldSlotsMin(std::span<const u64> values);
u64 FoldSlotsSaturatingAdd(std::span<const u64> values);
u64 FoldHashFixedBinaryTree(std::span<const u64> ordered_values);
FoldResult FoldOrderedSlots(std::span<const u64> values,
                            FoldOperation operation);
FoldResult FoldStrictOrderedSlots(std::span<const u64> values,
                                  FoldOperation operation,
                                  StrictFloatReductionPolicy policy);

} // namespace rund::kernel
