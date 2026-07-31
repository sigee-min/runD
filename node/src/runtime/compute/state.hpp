#pragma once

#include "../../compute/host.hpp"
#include "../../compute/run/state.hpp"
#include "../task/scheduler/state.hpp"
#include "../task/scheduler/state/model/wake.hpp"
#include "../task/scheduler/state/model/work.hpp"
#include "operation.hpp"
#include "slots.hpp"
#include "terminal.hpp"

#include <rund/session/trace.hpp>
#include <rund/task/handle/ref.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace rund::compute::detail {
class CompileService;
}

namespace rund::node::runtime_detail {
struct ComputeHostState;
}

namespace rund::node::compute_detail {

struct TaskState final {
  std::size_t slot = 0u;
  runtime_detail::ComputeHostState *host = nullptr;
  Operation operation{};
  mutable std::mutex mutex{};
  mutable std::condition_variable retired_cv{};
  task::Handle handle{};
  ::rund::detail::task::ResultRef completion{};
  ExternalWake wake{};
  compute::Status status{compute::Status::fail(compute::Reason::TaskInvalid)};
  compute::Stats stats{};
  std::optional<compute::Result<compute::detail::RunState>> job_result{};
  std::optional<node::accel::detail::PreparedPipelineEvidence>
      pipeline_evidence{};
  compute::detail::CpuPipelineSchedule pipeline_schedule{};
  std::atomic_bool cancel_requested{false};
  std::atomic_bool backend_submitted{false};
  std::atomic<std::uint8_t> completion_phase{0u};
  std::atomic<TerminalPhase> terminal_phase{TerminalPhase::Open};
  std::atomic_bool external_started{false};
  bool submitted = false;
  bool retiring = false;
  bool retired = false;
  std::uint32_t frame_bytes{};
};

} // namespace rund::node::compute_detail

namespace rund::node::runtime_detail {

enum class CompileAction : std::uint8_t {
  Stop,
  Close,
};

struct ComputeWorkerBatch;

struct ComputeWorkerSlot final {
  SchedulerWork work{};
  ComputeWorkerBatch *batch = nullptr;
  kernel::u32 worker{};
  kernel::u32 partitions{};
  bool failed = false;
};

struct ComputeWorkerBatch final {
  ComputeHostState *host = nullptr;
  std::size_t slot = 0u;
  const kernel::Partition *partitions = nullptr;
  kernel::u32 partition_count{};
  kernel::u32 workers{};
  kernel::WorkerTask task{};
  kernel::WorkerSubmission *submission = nullptr;
  kernel::WorkerStats *stats = nullptr;
  std::vector<ComputeWorkerSlot> slots{};
};

struct ComputeHostState final {
  using Signal = void (*)(void *, ::rund::TraceEvent,
                          ::rund::TraceCode) noexcept;
  using Emit = void (*)(void *, const compute::Status &,
                        const compute::Stats &) noexcept;
  using TaskControl =
      void (*)(const std::shared_ptr<ComputeHostState> &) noexcept;
  using CompileControl =
      void (*)(const std::shared_ptr<compute::detail::CompileService> &,
               CompileAction) noexcept;

  Scheduler scheduler{};
  std::mutex control{};
  std::mutex mutex{};
  std::condition_variable drained{};
  kernel::WorkerBackend worker_backend{};
  kernel::WorkerBackend async_worker_backend{};
  kernel::u32 workers{};
  std::uint32_t task_capacity{};
  compute::Compile compile_resources{};
  std::shared_ptr<compute::detail::CompileService> compile_service{};
  std::vector<kernel::u32> worker_capacity{};
  SlotSet task_slots{};
  SlotSet worker_batch_slots{};
  std::vector<std::unique_ptr<ComputeWorkerBatch>> worker_batches{};
  std::vector<std::unique_ptr<compute_detail::TaskState>> tasks{};
  std::size_t outstanding{};
  bool scope_active = false;
  bool configured = false;
  bool accepting = false;
  bool closed = false;
  compute::Reason reject_reason{compute::Reason::RuntimeNotRunning};
  void *signal_context = nullptr;
  Signal signal = nullptr;
  void *emit_context = nullptr;
  Emit emit = nullptr;
  TaskControl cancel_tasks = nullptr;
  TaskControl retire_tasks = nullptr;
  CompileControl control_compile = nullptr;
  bool terminalizing = false;
  bool terminal_closed = false;
};

void BindLifecycle(const std::shared_ptr<ComputeHostState> &host,
                   ComputeHostState::TaskControl cancel,
                   ComputeHostState::TaskControl retire) noexcept;
void StopAdmission(const std::shared_ptr<ComputeHostState> &host) noexcept;
void CloseHost(const std::shared_ptr<ComputeHostState> &host) noexcept;

} // namespace rund::node::runtime_detail
