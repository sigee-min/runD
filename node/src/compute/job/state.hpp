#pragma once

#include "../../array.hpp"
#include "../../accel/kernel/prepared.hpp"
#include "../../accel/kernel/scratch.hpp"
#include "../../accel/kernel/view.hpp"
#include "../program/state.hpp"
#include "../run/state.hpp"
#include "../type.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace rund::compute::detail {

struct JobState;
using JobCompletion = void (*)(void *, Result<RunState>) noexcept;

struct AccelRunSlot final {
  std::shared_ptr<JobState> state{};
  std::shared_ptr<void> lifetime{};
  JobCompletion completion = nullptr;
  void *user = nullptr;
};

enum class JobPhase : unsigned char {
  Idle,
  Queued,
  Running,
  Writing,
  Failed,
};

[[nodiscard]] constexpr bool job_busy(const JobPhase phase) noexcept {
  return phase == JobPhase::Queued || phase == JobPhase::Running ||
         phase == JobPhase::Writing;
}

[[nodiscard]] constexpr bool job_failed(const JobPhase phase) noexcept {
  return phase == JobPhase::Failed;
}

struct JobTerminalState final {
  std::optional<RunState> last{};
  std::optional<Stats> failed_stats{};
};

struct JobBufferView final {
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
};

// CPU maps consume strided views directly.  Reference collectives and
// primitives require dense ports, so only those external bindings receive a
// cold-prepared dense staging buffer.  The original owner/view are retained
// here for allocation-free gather/publish on every execution.
struct CpuViewTransfer final {
  std::shared_ptr<BufferState> external;
  JobBufferView view{};
  std::uint32_t binding{};
};

struct CpuViewTransferSlot final {
  std::uint32_t index{};
  std::uint64_t bytes{};

  [[nodiscard]] constexpr bool
  operator==(const CpuViewTransferSlot &) const noexcept = default;
};

// Program-intrinsic dense-port requirements. Computing this descriptor walks
// the immutable CPU runtime graph; applying it to one Pipeline route only
// projects that route's View extents and never walks the graph again.
struct CpuViewTransferRequirements final {
  const ProgramState *program = nullptr;
  std::uint64_t graph_hash{};
  std::size_t input_count{};
  std::size_t output_count{};
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;

  [[nodiscard]] bool
  operator==(const CpuViewTransferRequirements &) const noexcept = default;
};

// Cold planning resolves the Program's dense-port requirements once and
// freezes only the occurrence-specific strided bindings. Pipeline admission
// and Job materialization consume this same layout, so preparation cannot
// rediscover a larger transfer set after the budget check.
struct CpuViewTransferLayout final {
  const ProgramState *program = nullptr;
  std::uint64_t graph_hash{};
  std::size_t input_count{};
  std::size_t output_count{};
  std::vector<CpuViewTransferSlot> inputs;
  std::vector<CpuViewTransferSlot> outputs;
  std::uint64_t bytes{};

  [[nodiscard]] bool
  operator==(const CpuViewTransferLayout &) const noexcept = default;
};

// A recurrence owns one Program-internal value workspace for its complete
// lifetime.  Its occurrences have distinct external binding routes, but they
// execute serially behind the owning Pipeline gate and therefore share these
// dead-after-occurrence buffers without changing value or operation order.
struct JobWorkspace final {
  JobWorkspace() noexcept = default;
  JobWorkspace(const JobWorkspace &) = delete;
  JobWorkspace &operator=(const JobWorkspace &) = delete;
  JobWorkspace(JobWorkspace &&) = delete;
  JobWorkspace &operator=(JobWorkspace &&) = delete;

  std::shared_ptr<ProgramState> program;
  ::rund::node::detail::PreparedArray<std::shared_ptr<BufferState>> buffers;
  ::rund::node::detail::PreparedArray<std::size_t> offsets;
  std::shared_ptr<struct JobArena> arena;
};

struct JobArenaSlot final {
  std::size_t words{};
  std::size_t owner{};
  std::size_t offset_words{};
};

// Pipeline planning owns these dense View slots exactly once. Every private
// Job and recurrence phase borrows subranges of the same Buffer owners; a
// backend prepared command may bind them but cannot allocate a second scratch
// owner.
struct JobArena final {
  std::vector<std::shared_ptr<BufferState>> buffers;
  std::vector<JobArenaSlot> slots;
  node::accel::detail::RunBinds binds;
  node::accel::detail::KernelScratchLayout scratch;
  std::mutex gate;
  bool bound{};
};

struct JobState final {
  std::shared_ptr<ProgramState> program;
  std::shared_ptr<JobWorkspace> workspace;
  // Declared before every borrowed array so reverse member destruction drops
  // all views before releasing their sealed backing mapping.
  std::shared_ptr<CpuPreparedArena> cpu_prepared_arena;
  ::rund::node::detail::PreparedArray<std::shared_ptr<BufferState>> inputs;
  ::rund::node::detail::PreparedArray<JobBufferView> input_views;
  ::rund::node::detail::PreparedArray<std::shared_ptr<BufferState>>
      write_inputs;
  ::rund::node::detail::PreparedArray<std::shared_ptr<BufferState>>
      graph_buffers;
  ::rund::node::detail::PreparedArray<std::shared_ptr<BufferState>> outputs;
  ::rund::node::detail::PreparedArray<JobBufferView> output_views;
  node::accel::detail::KernelViewLayout views;
  ::rund::node::detail::PreparedArray<CpuViewTransfer> cpu_view_inputs;
  ::rund::node::detail::PreparedArray<CpuViewTransfer> cpu_view_outputs;
  std::uint64_t cpu_view_gather_bytes{};
  node::accel::detail::PreparedKernelRun prepared;
  node::accel::detail::PreparedKernelRun write_prepared;
  EmbeddedOwner<CpuRun> cpu;
  AccelRunSlot accel_run{};
  // Public Jobs retain result/read telemetry. Pipeline's private prepared Jobs
  // have no public Run receipt and intentionally leave this owner absent.
  std::unique_ptr<JobTerminalState> terminal;
  WriteStats write{};
  std::uint64_t transfer_peak{};
  std::uint64_t transfer_bytes{};
  std::uint64_t staging_peak{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_reused{};
  std::uint64_t staging_budget{};
  std::uint64_t frame_current{};
  std::uint64_t frame_peak{};
  std::uint64_t frame_bytes{};
  std::uint64_t frame_reused{};
  std::uint64_t frame_budget{};
  std::uint64_t run_count{};
  mutable std::mutex gate;
  JobPhase phase{JobPhase::Idle};
  Reason failure{Reason::ResidentNotRun};
};

[[nodiscard]] inline std::span<const std::shared_ptr<BufferState>>
job_graph_buffers(const JobState &state) noexcept {
  return state.workspace == nullptr
             ? std::span<const std::shared_ptr<BufferState>>{state
                                                                 .graph_buffers}
             : std::span<const std::shared_ptr<BufferState>>{
                   state.workspace->buffers};
}

[[nodiscard]] inline JobBufferView
job_whole_view(const BufferState &buffer) noexcept {
  const std::size_t bytes = type_bytes(buffer.type);
  return JobBufferView{.count = buffer.count,
                       .stride = 1u,
                       .element_bytes = bytes,
                       .alignment = bytes};
}

[[nodiscard]] inline JobBufferView
job_binding_view(const JobState &state, const GraphRunBinding &binding,
                 const BufferState &buffer) noexcept {
  if (state.program == nullptr ||
      binding.value_index >= state.program->graph_value_routes.size()) {
    return job_whole_view(buffer);
  }
  const GraphValueRoute route =
      state.program->graph_value_routes[binding.value_index];
  if (route.source == GraphBindSource::Internal && route.element_bytes != 0u &&
      route.index < state.program->chunks.size() &&
      route.count <= std::numeric_limits<std::size_t>::max()) {
    std::uint64_t offset_bytes = route.offset_bytes;
    if (state.workspace != nullptr) {
      if (route.index >= state.workspace->offsets.size() ||
          state.workspace->offsets[route.index] >
              std::numeric_limits<std::uint64_t>::max() /
                  sizeof(std::uint32_t)) {
        return {};
      }
      const std::uint64_t base =
          static_cast<std::uint64_t>(state.workspace->offsets[route.index]) *
          sizeof(std::uint32_t);
      if (route.offset_bytes >
          std::numeric_limits<std::uint64_t>::max() - base) {
        return {};
      }
      offset_bytes += base;
    }
    if (offset_bytes % route.element_bytes != 0u ||
        offset_bytes / route.element_bytes >
            std::numeric_limits<std::size_t>::max()) {
      return {};
    }
    return JobBufferView{
        .offset = static_cast<std::size_t>(offset_bytes / route.element_bytes),
        .count = static_cast<std::size_t>(route.count),
        .stride = 1u,
        .element_bytes = static_cast<std::size_t>(route.element_bytes),
        .alignment = static_cast<std::size_t>(route.alignment)};
  }
  if (route.source == GraphBindSource::Input &&
      route.index < state.input_views.size()) {
    return state.input_views[route.index];
  }
  if (route.source == GraphBindSource::Output &&
      route.index < state.output_views.size()) {
    return state.output_views[route.index];
  }
  return job_whole_view(buffer);
}

[[nodiscard]] inline JobBufferView
job_value_view(const JobState &state, const std::uint32_t value,
               const BufferState &buffer) noexcept {
  return value == 0u
             ? JobBufferView{}
             : job_binding_view(
                   state, GraphRunBinding{.value_index = value - 1u}, buffer);
}

[[nodiscard]] Status
prepare_job_accel(const std::shared_ptr<JobState> &state,
                  std::span<rund::AccelRunBinding> bindings);
[[nodiscard]] Result<RunState>
run_job_accel(const std::shared_ptr<JobState> &state);
[[nodiscard]] Result<RunState>
finish_job_accel(const std::shared_ptr<JobState> &state,
                 const rund::AccelEvidence &evidence);
[[nodiscard]] Status submit_job_accel(const std::shared_ptr<JobState> &state,
                                      std::shared_ptr<void> lifetime,
                                      JobCompletion completion,
                                      void *user) noexcept;
[[nodiscard]] Status finish_job(const std::shared_ptr<JobState> &state,
                                Result<RunState> result);
[[nodiscard]] Status queue_job(const std::shared_ptr<JobState> &state);
[[nodiscard]] Status cancel_job(const std::shared_ptr<JobState> &state);
[[nodiscard]] Status fail_job(const std::shared_ptr<JobState> &state,
                              Status failure);
void record_job_frame(const std::shared_ptr<JobState> &state,
                      std::uint64_t bytes, bool reused,
                      std::uint64_t budget) noexcept;
void release_job_frame(const std::shared_ptr<JobState> &state,
                       std::uint64_t bytes) noexcept;
[[nodiscard]] Status
read_job_buffer(const RunState &run, const std::shared_ptr<BufferState> &buffer,
                void *data, std::size_t bytes, std::size_t logical_count,
                std::uint64_t &staging_bytes, bool &staging_reused,
                std::uint64_t &staging_budget, bool destination_zeroed);

} // namespace rund::compute::detail
