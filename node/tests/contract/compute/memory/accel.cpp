#include "model.hpp"

#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>

#include "../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../src/accel/kernel/memory.hpp"
#include "../../../../src/accel/context/transfer.hpp"
#include "../../../../src/compute/device/state.hpp"
#include "../../../../src/compute/flow/state.hpp"
#include "../../../../src/compute/job/state.hpp"
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../src/accel/vulkan/buffer/create/telemetry.hpp"
#include "../../../../src/accel/vulkan/buffer/pool.hpp"
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace rund_node_memory_contract {

int CheckAccelMemory(const rund::compute::Backend backend) {
  const std::array<std::uint32_t, 4u> values{1u, 2u, 3u, 4u};
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device) {
    std::fprintf(stderr, "memory backend=%u open reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return 1;
  }
  {
    auto buffer = device->upload<std::uint32_t>(values);
    if (!buffer) {
      return 2;
    }
    const auto active = device->memory();
    if (!ValidStats(active) ||
        active.device.current < values.size() * sizeof(std::uint32_t) ||
        active.device.peak < active.device.current ||
        active.device.cumulative < active.device.current ||
        active.device.budget < active.device.current ||
        active.transfer.current != 0u ||
        active.transfer.cumulative != values.size() * sizeof(std::uint32_t) ||
        active.transfer.budget == 0u) {
      return 3;
    }
  }
  const auto released = device->memory();
  if (!ValidStats(released) || released.device.current != 0u ||
      released.device.peak == 0u) {
    return 4;
  }
  if (backend == rund::compute::Backend::Vulkan) {
    constexpr std::size_t staging_budget = 1024u * 1024u;
    constexpr std::size_t pool_limit = staging_budget * 32u;
    constexpr std::size_t count =
        staging_budget / sizeof(std::uint32_t) + 17u;
    std::vector<std::uint32_t> input(count);
    for (std::size_t index = 0u; index < input.size(); ++index) {
      input[index] = static_cast<std::uint32_t>(index * 17u + 3u);
    }
    auto buffer = device->upload<std::uint32_t>(input);
    const std::shared_ptr<rund::compute::detail::DeviceState> &device_state =
        rund::compute::detail::DeviceAccess::state(*device);
    rund::compute::detail::AccelDeviceState *const accel =
        device_state == nullptr
            ? nullptr
            : rund::compute::detail::accel_device(*device_state);
    const std::shared_ptr<rund::compute::detail::BufferState> buffer_state =
        buffer ? rund::compute::detail::BufferAccess::state(*buffer) : nullptr;
    rund::compute::detail::AccelBufferState *const native =
        buffer_state == nullptr
            ? nullptr
            : rund::compute::detail::accel_buffer(*buffer_state);
    std::vector<std::uint32_t> output(count);
    if (accel == nullptr || native == nullptr) {
      return 5;
    }
    const rund::node::accel::detail::AccelTransfer transfer =
        rund::node::accel::detail::DownloadAccelBufferMeasured(
            accel->context, native->buffer, output.data(),
            output.size() * sizeof(std::uint32_t), 0u, false);
    const auto memory = device->memory();
    if (!transfer.check.ok || output != input ||
        transfer.staging_peak_bytes == 0u ||
        transfer.staging_peak_bytes > staging_budget ||
        transfer.command_submits < 2u ||
        memory.staging.current > pool_limit ||
        memory.staging.peak > pool_limit + staging_budget) {
      return 6;
    }
  }
  return 0;
}

int CheckAccelProgramHostAccounting(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto program =
      on(*device)
          .input<std::uint32_t>(32u)
          .map("accel-owner-map", [](auto value) { return value + 1u; })
          .scan(Scan::InclusiveSum)
          .compile();
  if (!program) {
    return 2;
  }
  const auto &state = detail::FlowAccess::state(*program);
  if (state == nullptr || state->accel == nullptr ||
      state->cpu_graph != nullptr ||
      state->accel->kernel_token_host_bytes == 0u ||
      state->chunks.size() != 1u || state->graph_bindings.empty() ||
      state->graph_info.nodes.size() != 2u) {
    return 3;
  }
  MemoryStats memory{};
  if (!ReadMemory(*program, memory)) {
    return 4;
  }
  MemoryStats snapshot_stats{};
  const SnapshotAccounting snapshot = SnapshotMemory(*program, snapshot_stats);
  const PhysicalInternal physical = PhysicalInternalMemory(*program);
  const std::uint64_t owner_floor = sizeof(detail::ProgramState) +
                                    sizeof(detail::AccelProgram) +
                                    state->accel->kernel_token_host_bytes;
  if (!snapshot.complete || !snapshot.valid || !snapshot.allocation_free ||
      snapshot.metadata_entries != 1u || snapshot.tile_entries != 1u ||
      snapshot.internal_entries != 0u || !SameStats(memory, snapshot_stats) ||
      !physical.complete || physical.count != 0u || physical.bytes != 0u ||
      memory.host.current < owner_floor || memory.device.current != 0u ||
      memory.tile.current != 0u) {
    return 5;
  }
  return 0;
}

int CheckVulkanMemoryModel() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  const PreparedMemory prepared = VulkanPreparedMemory(
      VulkanMemoryStats{.current = 32u, .cumulative = 64u, .reused = 8u},
      VulkanMemoryStats{.current = 96u, .cumulative = 192u, .reused = 72u},
      1024u);
  if (prepared.current != 64u || prepared.peak != 64u ||
      prepared.cumulative != 128u || prepared.reused != 64u ||
      prepared.budget != 1024u) {
    return 1;
  }
  constexpr std::uint64_t staging_budget = 1024u * 1024u;
  constexpr std::uint64_t pool_limit =
      staging_budget * kVulkanPoolCapacity;
  if (VulkanPoolLimit(staging_budget) != pool_limit ||
      !FitsVulkanPool(pool_limit - 64u, 64u, pool_limit) ||
      FitsVulkanPool(pool_limit - 63u, 64u, pool_limit) ||
      FitsVulkanPool(0u, pool_limit + 1u, pool_limit)) {
    return 6;
  }
  VulkanAdapter physical{};
  physical.staging_memory.pooled = 96u;
  VulkanBuffer physical_buffer{.bytes = 32u};
  RecordVulkanMemoryLease(physical, physical_buffer, false,
                          VulkanMemoryUse::Staging);
  if (VulkanPhysicalStaging(physical) != 128u ||
      physical.staging_memory.peak != 128u) {
    return 7;
  }
  ReleaseVulkanMemoryLease(physical, physical_buffer);
  if (VulkanPhysicalStaging(physical) != 96u ||
      physical.staging_memory.peak != 128u) {
    return 8;
  }
  VulkanAdapter adapter{};
  VulkanBuffer buffer{.bytes = 64u};
  RecordVulkanMemoryLease(adapter, buffer, false, VulkanMemoryUse::Staging);
  if (adapter.staging_memory.current != 64u ||
      adapter.staging_memory.peak != 64u ||
      adapter.staging_memory.cumulative != 64u ||
      adapter.staging_memory.reused != 0u) {
    return 2;
  }
  ReleaseVulkanMemoryLease(adapter, buffer);
  RecordVulkanMemoryLease(adapter, buffer, true, VulkanMemoryUse::Staging);
  ReleaseVulkanMemoryLease(adapter, buffer);
  if (adapter.staging_memory.current != 0u ||
      adapter.staging_memory.cumulative != 128u ||
      adapter.staging_memory.reused != 64u) {
    return 3;
  }
  RecordVulkanMemoryLease(adapter, buffer, true, VulkanMemoryUse::Resident);
  if (buffer.memory_lease || adapter.staging_memory.cumulative != 128u) {
    return 4;
  }
  adapter.staging_memory = VulkanMemoryStats{.current = kCounterMaximum - 1u,
                                             .peak = kCounterMaximum - 1u,
                                             .cumulative = kCounterMaximum - 1u,
                                             .reused = kCounterMaximum - 1u};
  buffer.bytes = 2u;
  RecordVulkanMemoryLease(adapter, buffer, true, VulkanMemoryUse::Staging);
  ReleaseVulkanMemoryLease(adapter, buffer);
  if (adapter.staging_memory.current != kCounterMaximum ||
      adapter.staging_memory.peak != kCounterMaximum ||
      adapter.staging_memory.cumulative != kCounterMaximum ||
      adapter.staging_memory.reused != kCounterMaximum) {
    return 5;
  }
#endif
  return 0;
}

int CheckRetainedJobMemory(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 4u> input{1u, 2u, 3u, 4u};
  auto program = on(rund::node::test_contract::target_for(backend))
                     .map<std::uint32_t>("memory-copy", input.size(),
                                         [](auto value) { return value; })
                     .scan(Scan::InclusiveSum)
                     .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }
  const std::shared_ptr<detail::JobState> state =
      detail::JobAccess::state(*job);
  if (state == nullptr || state->program == nullptr ||
      state->program->accel == nullptr ||
      state->graph_buffers.size() != state->program->chunks.size() ||
      !state->prepared.ok) {
    return 11;
  }
  node_compute_allocation::Start();
  const MemoryStats resident = job->memory();
  const MemoryCounter prepared = resident.staging;
  node_compute_allocation::Stop();
  if (resident.transfer.current != 0u ||
      resident.transfer.cumulative != input.size() * sizeof(std::uint32_t) ||
      prepared.current == 0u || prepared.peak != prepared.current ||
      prepared.cumulative < prepared.current ||
      prepared.budget < prepared.current ||
      node_compute_allocation::Count() != 0u) {
    return 3;
  }
  if (!job->run()) {
    return 4;
  }
  const MemoryCounter warm = job->memory().staging;
  if (warm.current != prepared.current || warm.peak != prepared.peak ||
      warm.cumulative != prepared.cumulative ||
      warm.reused != prepared.reused) {
    return 5;
  }
  std::array<MemoryEntry, 32u> entries{};
  const MemorySnapshot snapshot = job->memory_snapshot(entries);
  bool found = false;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    found = found || (entries[index].category == MemoryCategory::Staging &&
                      entries[index].bytes.current == prepared.current);
  }
  if (!found) {
    return 6;
  }
  return 0;
}

int CheckSortRunMemory(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 4u> input{3u, 1u, 4u, 2u};
  auto program = on(rund::node::test_contract::target_for(backend))
                     .map<std::uint32_t>("memory-sort", input.size(),
                                         [](auto value) { return value; })
                     .sort()
                     .compile();
  if (!program) {
    std::fprintf(stderr, "sort memory backend=%u compile=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr, "sort memory backend=%u prepare=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  const bool retained = backend == Backend::Metal || backend == Backend::Vulkan;
  const MemoryCounter prepared = job->memory().staging;
  if (!job->run()) {
    return 3;
  }
  const MemoryCounter first = job->memory().staging;
  if (first.current != prepared.current ||
      (retained ? (prepared.current == 0u || first.peak != prepared.peak ||
                   first.cumulative != prepared.cumulative)
                : (first.peak < prepared.peak || first.peak <= first.current ||
                   first.cumulative <= prepared.cumulative))) {
    std::fprintf(stderr,
                 "sort memory backend=%u prepared=%llu/%llu/%llu "
                 "first=%llu/%llu/%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(prepared.current),
                 static_cast<unsigned long long>(prepared.peak),
                 static_cast<unsigned long long>(prepared.cumulative),
                 static_cast<unsigned long long>(first.current),
                 static_cast<unsigned long long>(first.peak),
                 static_cast<unsigned long long>(first.cumulative));
    return 4;
  }
  if (!job->run()) {
    return 5;
  }
  const MemoryCounter second = job->memory().staging;
  if (second.current != first.current ||
      (retained ? (second.cumulative != first.cumulative ||
                   second.reused != first.reused)
                : (second.peak != first.peak ||
                   second.cumulative <= first.cumulative ||
                   second.reused <= first.reused))) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_memory_contract
