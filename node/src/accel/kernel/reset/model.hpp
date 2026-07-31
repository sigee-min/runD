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
struct Result;
[[nodiscard]] Result Prove(Spec, std::uint64_t, std::uint64_t) noexcept;

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

private:
  constexpr explicit Range(const Spec spec) noexcept
      : offset_{spec.offset}, count_{spec.count}, stride_{spec.stride},
        element_{spec.element} {}

  std::uint64_t offset_{};
  std::uint64_t count_{};
  std::uint64_t stride_{};
  std::uint64_t element_{};

  friend Result Prove(Spec, std::uint64_t, std::uint64_t) noexcept;
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
