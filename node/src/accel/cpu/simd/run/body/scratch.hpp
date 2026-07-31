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
  const std::size_t nodes = prepared.instructions.size();
  const std::size_t values = nodes + 1u;
  constexpr std::size_t padding =
      alignof(ValueVec) + alignof(WideScalar) + alignof(std::uint8_t);
  return values * sizeof(std::uint8_t) + values * sizeof(ValueVec) +
         values * kLaneCount * sizeof(WideScalar) + padding;
}

struct RunScratch final {
  ValueVec *values = nullptr;
  WideScalar *wide = nullptr;
  std::uint8_t *wide_valid = nullptr;

  [[nodiscard]] explicit operator bool() const noexcept {
    return values != nullptr && wide != nullptr && wide_valid != nullptr;
  }
};

[[nodiscard]] RunScratch
PrepareRunScratch(const PreparedRun &prepared,
                  const CpuSimdScratch scratch) noexcept {
  if (scratch.bytes < RequiredScratchBytes(prepared)) {
    return {};
  }
  const std::size_t count = prepared.instructions.size();
  ScratchLayout layout(scratch);
  return RunScratch{
      .values = layout.take<ValueVec>(count + 1u),
      .wide = layout.take<WideScalar>((count + 1u) * kLaneCount),
      .wide_valid = layout.take<std::uint8_t>(count + 1u),
  };
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
