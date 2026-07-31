#pragma once

#include <rund/reason.hpp>
#include <rund/task/callable.hpp>
#include <rund/task/operation/kind.hpp>

#include "../../task/completion/model.hpp"

#include <coroutine>
#include <cstdint>

namespace rund::detail::task {
struct CoroutineOps;
}

namespace rund::node {

enum class TaskState : std::uint8_t {
  Ready,
  Running,
  Sleeping,
  JoinBlocked,
  ChannelBlocked,
  IoBlocked,
  ExternalBlocked,
  Completed,
  Failed,
};

struct TaskRecord {
  std::uint64_t id = 0u;
  std::uint64_t scope_id = 1u;
  std::uint64_t dynamic_scope_id = 1u;
  union {
    ::rund::detail::task::Callable *callable = nullptr;
    std::coroutine_handle<> coroutine_frame;
  };
  const ::rund::detail::task::CoroutineOps *coroutine_ops = nullptr;
  CompletionSlot completion{};
  // A task is either parked on a logical wait or queued for direct lane
  // dispatch. The two representations never have simultaneous authority.
  union {
    std::uint64_t wait_id = 0u;
    std::uint64_t wake_ticket;
  };
  union {
    std::uint64_t wait_source_id = 0u;
    std::uint64_t wait_token;
  };
  TaskRecord *wake_next = nullptr;
  std::uint32_t home_lane = 0u;
  ReasonCode failure_code = ReasonCode::Ok;
  ReasonCode wait_result = ReasonCode::Ok;
  ReasonCode io_result = ReasonCode::Ok;
  short io_revents = 0;
  TaskState state = TaskState::Ready;
  std::uint16_t coroutine_task : 1 = false;
  std::uint16_t coroutine_parked : 1 = false;
  std::uint16_t quantum_active : 1 = false;
  std::uint16_t lane_segment_side_exit : 1 = false;
  std::uint16_t resource_live : 1 = false;
  std::uint16_t recyclable : 1 = false;
};

static_assert(sizeof(void *) != 8u || sizeof(TaskRecord) <= 96u,
              "64-bit TaskRecord must remain at most six 16-byte lanes");

} // namespace rund::node
