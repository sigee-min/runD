#pragma once

#include <cstdint>

namespace rund::node {

struct TaskLane;
struct TaskRecord;

struct LaneBatchSubmission final {
  TaskLane *lane = nullptr;
  TaskRecord *record = nullptr;
  std::uint64_t task_id = 0u;
  std::uint64_t ticket = 0u;
  std::uint64_t sequence = 0u;
  bool split_primitive_packets = false;
  bool notify_ready = false;
};

} // namespace rund::node
