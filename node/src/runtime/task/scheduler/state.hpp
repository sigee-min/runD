#pragma once
#include <kernel/program/executor/model.hpp>
#include <node/runtime/replay/host/payload.hpp>
#include <node/runtime/replay/mode.hpp>
#include <rund/host/event.hpp>
#include <rund/net/socket.hpp>
#include <rund/replay/storage.hpp>
#include <rund/session/memory.hpp>
#include <rund/session/scheduler.hpp>
#include <rund/task/active.hpp>
#include <rund/task/callable.hpp>
#include <rund/task/cancel.hpp>
#include <rund/task/channel/access.hpp>
#include <rund/task/handle.hpp>
#include <rund/task/observation.hpp>
#include <rund/task/operation/kind.hpp>
#include <rund/task/results.hpp>
#include <rund/task/stats/storage.hpp>
#include <rund/task/status.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "../../../host/net/registry/access.hpp"
#include "../../../host/net/registry/socket.hpp"
#include "../../replay/host/payload/store.hpp"
#include "../../replay/scope/plan.hpp"
#include "hash.hpp"
#include "host.hpp"
#include "reactor/many.hpp"
#include "reactor/model.hpp"
#include "reactor/ready/set/model.hpp"
#include "state/model/batch.hpp"
#include "state/model/join.hpp"
#include "state/model/stop.hpp"
#include "state/model/task.hpp"
#include "state/model/timer.hpp"
#include "state/segment.hpp"
#include "task/callable/pool.hpp"
#include "task/completion/model.hpp"
#include "task/frame.hpp"

namespace rund {
struct ReplayConfig;
}

namespace rund::task {
template <typename T> class Task;
}

namespace rund::detail::task {
class ApiAccess;
struct AwaitDecision;
struct CoroutineStart;
struct IoDecision;
struct ResultRef;
struct Spawned;
} // namespace rund::detail::task

namespace rund::net {
struct Limits;

namespace ready {
struct Config;
struct Event;
struct Request;
struct Set;
struct Status;

namespace many {
struct Budget;
class Wait;
} // namespace many
} // namespace ready
} // namespace rund::net

namespace rund::host::io {
struct CloseResult;
class Fd;
class FdView;
class ReadOp;
struct ReadResult;
class WriteOp;
struct WriteResult;
} // namespace rund::host::io

namespace rund::node {

#include "state/fail.hpp"
#include "state/forward.hpp"
#include "state/reactor/cleanup.hpp"
#include "state/scope.hpp"

struct SchedulerState;

class Scheduler {
public:
#include "state/channel/public.hpp"
#include "state/core/public.hpp"
#include "state/host/public.hpp"
#include "state/reactor/public.hpp"
#include "state/runtime/public.hpp"
#include "state/stop/public.hpp"
#include "state/task/public.hpp"

private:
#include "state/friends.hpp"

#include "state/host.hpp"
#include "state/lane.hpp"
#include "state/progress.hpp"
#include "state/reactor/timeout.hpp"
#include "state/record.hpp"
#include "state/reset.hpp"
#include "state/spawn.hpp"
#include "state/task.hpp"

  SchedulerState *state_ = nullptr;
  std::vector<LaneOwnedSegmentLane> lane_segments_{};
  std::vector<std::uint64_t> lane_executed_{};
  std::vector<LaneSegmentEffect> lane_effects_{};
  static thread_local Scheduler *active_;
  static std::atomic<std::uint64_t> next_scheduler_id_;
};

} // namespace rund::node
