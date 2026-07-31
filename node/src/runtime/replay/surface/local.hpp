#pragma once

#include "data.hpp"

#include <rund/replay/run.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::replay {

namespace detail::surface {

[[nodiscard]] constexpr std::uint64_t count(const std::size_t value) noexcept {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(value);
}

void publish(scope::Timing &timing, telemetry::Event &event,
             Session::Result *result = nullptr) noexcept;

[[nodiscard]] Code restore(const Checkpoint &checkpoint,
                           RestoreCall restore) noexcept;

} // namespace detail::surface
} // namespace rund::replay
