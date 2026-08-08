#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::cpu_simd_detail {
namespace {

class ScratchLayout {
public:
  explicit ScratchLayout(const CpuSimdScratch scratch) noexcept
      : current_(static_cast<std::byte *>(scratch.data)),
        end_(current_ == nullptr ? nullptr : current_ + scratch.bytes) {}

  template <class T> [[nodiscard]] T *take(const std::size_t count) noexcept {
    if (current_ == nullptr || end_ == nullptr) {
      return nullptr;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(current_);
    const auto aligned = (address + alignof(T) - 1u) & ~(alignof(T) - 1u);
    auto *const start = reinterpret_cast<std::byte *>(aligned);
    if (start > end_ ||
        count > static_cast<std::size_t>(end_ - start) / sizeof(T)) {
      return nullptr;
    }
    current_ = start + count * sizeof(T);
    return reinterpret_cast<T *>(start);
  }

private:
  std::byte *current_ = nullptr;
  std::byte *end_ = nullptr;
};

[[nodiscard]] std::size_t
RequiredScratchBytes(const PreparedRun &prepared) noexcept {
  static_assert(alignof(WideScalar) <= alignof(ValueVec));
  static_assert(sizeof(ValueVec) % alignof(WideScalar) == 0u);
  const std::size_t values = prepared.value_slot_count;
  const bool fixed = prepared.domain == rund::kernel::ComputeDomain::Fixed;
  const std::size_t padding = alignof(ValueVec) - 1u;
  return values * sizeof(ValueVec) +
         (fixed ? values * kLaneCount * sizeof(WideScalar) +
                      values * sizeof(std::uint8_t)
                : 0u) +
         padding;
}

struct RunScratch final {
  ValueVec *values = nullptr;
  WideScalar *wide = nullptr;
  std::uint8_t *wide_valid = nullptr;
  bool wide_required = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return values != nullptr &&
           (!wide_required || (wide != nullptr && wide_valid != nullptr));
  }
};

[[nodiscard]] RunScratch
PrepareRunScratch(const PreparedRun &prepared,
                  const CpuSimdScratch scratch) noexcept {
  if (scratch.bytes < RequiredScratchBytes(prepared)) {
    return {};
  }
  const std::size_t count = prepared.value_slot_count;
  const bool fixed = prepared.domain == rund::kernel::ComputeDomain::Fixed;
  ScratchLayout layout(scratch);
  return RunScratch{
      .values = layout.take<ValueVec>(count),
      .wide = fixed ? layout.take<WideScalar>(count * kLaneCount) : nullptr,
      .wide_valid = fixed ? layout.take<std::uint8_t>(count) : nullptr,
      .wide_required = fixed,
  };
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
