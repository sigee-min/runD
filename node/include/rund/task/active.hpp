#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::detail::task {

struct ActiveState {
  bool active = false;
  std::uint64_t scheduler_id = 0u;
  std::uint64_t task_id = 0u;
  std::size_t task_slot = std::numeric_limits<std::size_t>::max();
  std::size_t task_capacity = 0u;
};

} // namespace rund::detail::task
