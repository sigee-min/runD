#include "../backend.hpp"
#include "../device/info.hpp"
#include "../job/state.hpp"
#include "cpu.hpp"
#include "local.hpp"
#include "profile.hpp"

#include <rund/counter.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::span<const std::shared_ptr<BufferState>>
internal_buffers(const JobState &state) noexcept {
  if (state.workspace != nullptr) {
    return state.workspace->buffers;
  }
  if (state.cpu != nullptr && state.cpu->graph != nullptr) {
    return state.cpu->graph->buffers;
  }
  return state.graph_buffers;
}

[[nodiscard]] std::uint64_t host_memory(const JobState &state,
                                        const std::uint64_t cpu_host) noexcept {
  std::uint64_t bytes = sizeof(JobState);
  if (state.terminal != nullptr) {
    bytes = add_cpu_memory_bytes(bytes, sizeof(JobTerminalState));
  }
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.inputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.input_views));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.write_inputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_buffers));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.outputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.output_views));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.cpu_view_inputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.cpu_view_outputs));
  if (state.workspace != nullptr) {
    const bool arena_workspace = state.workspace->buffers.borrowed() &&
                                 state.workspace->offsets.borrowed();
    if (!arena_workspace) {
      bytes = add_cpu_memory_bytes(bytes, sizeof(JobWorkspace));
    }
    bytes =
        add_cpu_memory_bytes(bytes, vector_memory(state.workspace->buffers));
    bytes =
        add_cpu_memory_bytes(bytes, vector_memory(state.workspace->offsets));
  }
  return add_cpu_memory_bytes(bytes, cpu_host);
}

void add_buffers(SnapshotWriter *const writer, BufferMemory &total,
                 const std::span<const std::shared_ptr<BufferState>> buffers,
                 const MemoryUse use, const Backend backend) noexcept {
  const MemoryCategory physical =
      backend == Backend::Cpu ? MemoryCategory::Host : MemoryCategory::Device;
  for (std::uint32_t index = 0u; index < buffers.size(); ++index) {
    if (std::find(buffers.begin(), buffers.begin() + index, buffers[index]) !=
        buffers.begin() + index) {
      continue;
    }
    const BufferMemory memory = measure_buffer(buffers[index]);
    add_buffer_memory(total, memory);
    if (writer != nullptr) {
      writer->add(MemoryCategory::Resident, use, index,
                  fixed_memory(memory.resident));
      writer->add(physical, use, index,
                  fixed_memory(memory.physical, memory.reused));
    }
  }
}

struct JobMemoryView final {
  MemoryStats stats{};
};

[[nodiscard]] JobMemoryView
memory_view(const JobState &state,
            SnapshotWriter *const writer = nullptr) noexcept {
  const auto &internal = internal_buffers(state);
  const CpuRetainedMemory cpu = cpu_run_memory(state.cpu.get());
  const std::uint64_t host = host_memory(state, cpu.host);
  const Backend backend = memory_backend(*state.program->device);
  if (writer != nullptr) {
    writer->add(MemoryCategory::Host, MemoryUse::Metadata, 0u,
                fixed_memory(host));
  }
  BufferMemory buffers{};
  add_buffers(writer, buffers, state.inputs, MemoryUse::Input, backend);
  add_buffers(writer, buffers, state.write_inputs, MemoryUse::PendingInput,
              backend);
  add_buffers(writer, buffers, internal, MemoryUse::Internal, backend);
  add_buffers(writer, buffers, state.outputs, MemoryUse::Output, backend);
  const std::uint64_t tile = cpu.tile;
  const std::uint64_t tile_reused =
      state.run_count == 0u || tile == 0u
          ? 0u
          : (state.run_count > std::numeric_limits<std::uint64_t>::max() / tile
                 ? std::numeric_limits<std::uint64_t>::max()
                 : state.run_count * tile);
  MemoryCounter staging{};
  const DeviceState &device = *state.program->device;
  if (device.ops != nullptr && device.ops->job_staging != nullptr) {
    staging = device.ops->job_staging(state);
  }
  staging.cumulative = ::rund::detail::counter::SaturatingAdd(
      staging.cumulative, state.staging_bytes);
  staging.reused = ::rund::detail::counter::SaturatingAdd(staging.reused,
                                                          state.staging_reused);
  staging.budget = std::max(staging.budget, state.staging_budget);
  const std::uint64_t read_peak = ::rund::detail::counter::SaturatingAdd(
      staging.current, state.staging_peak);
  staging.peak = std::max(staging.peak, read_peak);
  MemoryStats stats{
      .backend = backend,
      .scope = MemoryScope::Job,
      .host = fixed_memory(host),
      .frame = MemoryCounter{.current = state.frame_current,
                             .peak = state.frame_peak,
                             .cumulative = state.frame_bytes,
                             .reused = state.frame_reused,
                             .budget = state.frame_budget},
      .tile = fixed_memory(tile, tile_reused),
      .resident = fixed_memory(buffers.resident),
      .staging = MemoryCounter{.current = staging.current,
                               .peak = staging.peak,
                               .cumulative = staging.cumulative,
                               .reused = staging.reused,
                               .budget = staging.budget},
      .transfer = MemoryCounter{.peak = state.transfer_peak,
                                .cumulative = state.transfer_bytes,
                                .budget = device_budget(state.program->device)},
  };
  set_physical(stats, buffers.physical, buffers.reused);
  if (writer != nullptr) {
    writer->set_summary(stats);
    writer->add(MemoryCategory::Frame, MemoryUse::Coordinator, 0u, stats.frame);
    writer->add(MemoryCategory::Tile, MemoryUse::Scratch, 0u, stats.tile);
    writer->add(MemoryCategory::Staging, MemoryUse::Scratch, 0u, stats.staging);
    writer->add(MemoryCategory::Transfer, MemoryUse::Traffic, 0u,
                stats.transfer);
  }
  return JobMemoryView{.stats = stats};
}

} // namespace

MemoryStats job_memory(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return memory_view(*state).stats;
}

Result<telemetry::Profile>
job_profile(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::ProfileInvalid);
  }
  auto device = snapshot_device_info(state->program->device);
  if (!device) {
    return Result<telemetry::Profile>::fail(device.reason());
  }
  std::lock_guard lock{state->gate};
  if (job_busy(state->phase)) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  const Backend backend = state->program->device->backend;
  Stats execution{.backend = backend};
  if (state->terminal != nullptr && state->terminal->last.has_value()) {
    execution = run_stats(*state->terminal->last);
  } else if (state->terminal != nullptr &&
             state->terminal->failed_stats.has_value()) {
    execution = *state->terminal->failed_stats;
  }
  return Result<telemetry::Profile>::success(ProfileAccess::make(
      std::move(*device), execution, memory_view(*state).stats));
}

MemorySnapshot
job_memory_snapshot(const std::shared_ptr<JobState> &state,
                    const std::span<MemoryEntry> entries) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr) {
    SnapshotWriter writer{MemoryStats{}, entries};
    return writer.finish();
  }
  std::lock_guard lock{state->gate};
  SnapshotWriter writer{MemoryStats{}, entries};
  (void)memory_view(*state, &writer);
  return writer.finish();
}

} // namespace rund::compute::detail
