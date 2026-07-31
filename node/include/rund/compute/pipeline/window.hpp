#pragma once

#include <rund/compute/pipeline/bind.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace rund::compute {

namespace detail {
struct WindowAccess;
}

inline constexpr std::size_t NoWindowTerminal =
    std::numeric_limits<std::size_t>::max();

template <std::size_t Terminal = NoWindowTerminal> class WindowInput final {
public:
  static constexpr std::size_t terminal_index = Terminal;

  template <std::size_t Index>
  [[nodiscard]] WindowInput<Index>
  until(const std::uint32_t expected = 1u) const noexcept {
    return WindowInput<Index>{count_, expected};
  }

private:
  friend class PipelineBuilder;
  friend struct detail::WindowAccess;
  template <std::size_t> friend class WindowInput;

  explicit WindowInput(detail::ResourceView count,
                       const std::uint32_t expected = 1u) noexcept
      : count_(std::move(count)), expected_(expected) {}

  detail::ResourceView count_{};
  std::uint32_t expected_{1u};
};

namespace detail {

struct WindowAccess final {
  [[nodiscard]] static WindowInput<> make(ResourceView count) noexcept {
    return WindowInput<>{std::move(count)};
  }
};

} // namespace detail

template <class Count>
  requires std::is_same_v<std::remove_const_t<Count>, std::uint32_t>
[[nodiscard]] WindowInput<> window(const Buffer<Count> &count) noexcept {
  return detail::WindowAccess::make(
      detail::BufferAccess::view(count, detail::ResourceAccess::Read));
}

template <class Count>
  requires std::is_same_v<std::remove_const_t<Count>, std::uint32_t>
[[nodiscard]] WindowInput<> window(const View<Count> &count) noexcept {
  return detail::WindowAccess::make(
      detail::BufferAccess::view(count, detail::ResourceAccess::Read));
}

} // namespace rund::compute
