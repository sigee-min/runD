#pragma once

#include <cstdint>

namespace rund::node {

struct JoinWait {
  std::uint64_t waiter_task_id = 0u;
  std::uint64_t target_task_id = 0u;
  std::uint64_t wait_id = 0u;
};

} // namespace rund::node
