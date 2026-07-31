#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr bool SegmentHeadValid(const u32 head,
                                              const u64 index) noexcept {
  return head <= 1u && (index != 0u || head == 1u);
}

[[nodiscard]] inline bool SegmentHeadsValid(const u32 *const heads,
                                            const u64 begin,
                                            const u64 count) noexcept {
  if (heads == nullptr || begin > count) {
    return false;
  }
  for (u64 index = begin; index < count; ++index) {
    if (!SegmentHeadValid(heads[index], index)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::kernel
