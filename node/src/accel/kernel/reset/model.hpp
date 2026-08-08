#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rund::node::accel::detail::reset {

inline constexpr std::uint64_t kWordBytes = sizeof(std::uint32_t);

struct Spec final {
  std::uint64_t offset{};
  std::uint64_t count{};
  std::uint64_t stride{};
  std::uint64_t element{};

  [[nodiscard]] constexpr bool dense() const noexcept {
    return stride == element;
  }
};

class Range;
[[nodiscard]] Range Prove(Spec, std::uint64_t) noexcept;

class Range final {
public:
  constexpr Range() noexcept = default;

  [[nodiscard]] constexpr std::uint64_t offset() const noexcept {
    return offset_;
  }

  [[nodiscard]] constexpr std::uint64_t count() const noexcept {
    return count_;
  }

  [[nodiscard]] constexpr std::uint64_t stride() const noexcept {
    return stride_;
  }

  [[nodiscard]] constexpr std::uint64_t element() const noexcept {
    return element_;
  }

  [[nodiscard]] constexpr bool dense() const noexcept {
    return stride_ == element_;
  }

  [[nodiscard]] constexpr std::uint64_t end() const noexcept { return end_; }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return count_ != 0u && end_ > offset_;
  }

private:
  constexpr Range(const Spec spec, const std::uint64_t end) noexcept
      : offset_{spec.offset}, count_{spec.count}, stride_{spec.stride},
        element_{spec.element}, end_{end} {}

  std::uint64_t offset_{};
  std::uint64_t count_{};
  std::uint64_t stride_{};
  std::uint64_t element_{};
  std::uint64_t end_{};

  friend Range Prove(Spec, std::uint64_t) noexcept;
};

struct Replacement final {
  std::uint64_t count{};
  std::uint64_t element{};
};

struct Params final {
  std::uint64_t count{};
  std::uint64_t base{};
  std::uint64_t offset_words{};
  std::uint64_t stride_words{};
  std::uint32_t element_words{};
  std::uint32_t reserved{};
};

static_assert(std::is_standard_layout_v<Params>);
static_assert(sizeof(Params) == 40u);
static_assert(offsetof(Params, count) == 0u);
static_assert(offsetof(Params, base) == 8u);
static_assert(offsetof(Params, offset_words) == 16u);
static_assert(offsetof(Params, stride_words) == 24u);
static_assert(offsetof(Params, element_words) == 32u);
static_assert(offsetof(Params, reserved) == 36u);

[[nodiscard]] constexpr std::uint64_t Payload(const Range range) noexcept {
  return range.count() * range.element();
}

[[nodiscard]] constexpr bool
WordAddressable(const Range range, const std::uint64_t origin,
                const std::uint64_t word_limit) noexcept {
  return range.valid() && origin <= range.offset() &&
         origin % kWordBytes == 0u &&
         (range.offset() - origin) % kWordBytes == 0u &&
         (range.end() - origin) / kWordBytes <= word_limit;
}

[[nodiscard]] constexpr std::uint64_t
Commands(const std::uint64_t count, const std::uint64_t window) noexcept {
  return window == 0u ? 0u
                      : count / window +
                            static_cast<std::uint64_t>(count % window != 0u);
}

[[nodiscard]] constexpr Params Bind(const Range range,
                                    const std::uint64_t base,
                                    const std::uint64_t origin = 0u) noexcept {
  return Params{
      .count = range.count(),
      .base = base,
      .offset_words = (range.offset() - origin) / kWordBytes,
      .stride_words = range.stride() / kWordBytes,
      .element_words = static_cast<std::uint32_t>(range.element() / kWordBytes),
  };
}

} // namespace rund::node::accel::detail::reset
