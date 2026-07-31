#pragma once

#include <rund/session/scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::runtime_detail {

[[nodiscard]] constexpr ::rund::SchedulerConfig NormalizeScheduler(
    ::rund::SchedulerConfig config,
    const std::uint32_t runtime_workers,
    const std::size_t trace_capacity) noexcept {
  if (config.task_workers == 0u) {
    config.task_workers = std::max<std::uint32_t>(1u, runtime_workers);
  }
  if (config.observation_capacity == 0u) {
    config.observation_capacity =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            trace_capacity, std::numeric_limits<std::uint32_t>::max()));
  }
  return config;
}

static_assert(NormalizeScheduler({}, 7u, 11u).task_workers == 7u);
static_assert(NormalizeScheduler({}, 7u, 11u).observation_capacity == 11u);
static_assert(NormalizeScheduler({}, 0u, 11u).task_workers == 1u);
static_assert(NormalizeScheduler({.task_workers = 3u,
                                  .observation_capacity = 5u},
                                 7u, std::numeric_limits<std::size_t>::max())
                  .task_workers == 3u);
static_assert(NormalizeScheduler({.task_workers = 3u,
                                  .observation_capacity = 5u},
                                 7u, std::numeric_limits<std::size_t>::max())
                  .observation_capacity == 5u);
static_assert(NormalizeScheduler({}, 7u,
                                 std::numeric_limits<std::size_t>::max())
                  .observation_capacity ==
              std::numeric_limits<std::uint32_t>::max());

}  // namespace rund::node::runtime_detail
