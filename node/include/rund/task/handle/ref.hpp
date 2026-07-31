#pragma once

#include <rund/task/status.hpp>

#include <cstdint>

namespace rund::detail::task {

struct ResultRef final {
  void *authority{};
  std::uint32_t slot{};
  std::uint32_t generation{};
  ::rund::task::Poll (*poll)(void *, std::uint32_t, std::uint32_t) noexcept {};
  ::rund::task::Status (*wait)(void *, std::uint32_t,
                               std::uint32_t) noexcept {};
  ::rund::task::Status (*copy)(void *, std::uint32_t, std::uint32_t, void *,
                               const void *, void (*)(void *, const void *)){};
  void (*release)(void *, std::uint32_t, std::uint32_t) noexcept {};
};

} // namespace rund::detail::task
