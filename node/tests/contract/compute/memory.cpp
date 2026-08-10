#include "memory/local.hpp"

#include "../target/selection.hpp"
#include "allocation.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

int RunComputeMemoryContract() {
  using rund::compute::Backend;
  using rund::compute::MemoryScope;

  if (!rund_node_memory_contract::CheckCounterSaturation()) {
    return 390;
  }
  if (!rund_node_memory_contract::CheckPreparedMemorySnapshot()) {
    return 391;
  }
  if (!rund_node_memory_contract::CheckTrafficMeterSnapshot()) {
    return 392;
  }
  if (const int scratch = rund_node_memory_contract::
          CheckAcceleratorScratchPlacementAuthority();
      scratch != 0) {
    return 350 + scratch;
  }

  if (const int arena = rund_node_memory_contract::CheckValueRouteArena();
      arena != 0) {
    return 400 + arena;
  }
  if (const int arena = rund_node_memory_contract::CheckCpuSealedArena();
      arena != 0) {
    return 410 + arena;
  }

  auto device = rund::compute::open(rund::compute::Target::cpu(2u));
  if (!device) {
    return 1;
  }
  {
    auto buffer = device->buffer<std::int32_t>(4u);
    if (!buffer) {
      return 12;
    }
    const auto active = device->memory();
    std::array<rund::compute::MemoryEntry, 7u> device_entries{};
    const auto device_snapshot = device->memory_snapshot(device_entries);
    bool found_logical = false;
    bool found_committed = false;
    for (std::size_t index = 0u; index < device_snapshot.written; ++index) {
      const rund::compute::MemoryEntry &entry = device_entries[index];
      found_logical =
          found_logical ||
          (entry.category == rund::compute::MemoryCategory::Resident &&
           entry.use == rund::compute::MemoryUse::Internal &&
           entry.bytes.current == 16u);
      found_committed =
          found_committed ||
          (entry.category == rund::compute::MemoryCategory::Host &&
           entry.use == rund::compute::MemoryUse::Internal &&
           entry.bytes.current == 16u);
    }
    if (active.resident.current != 16u || active.resident.peak != 16u ||
        active.resident.cumulative != 16u || active.host.current != 16u ||
        active.host.peak != 16u || active.host.cumulative != 16u ||
        device_snapshot.truncated() ||
        device_snapshot.summary.resident.current != active.resident.current ||
        device_snapshot.summary.host.current != active.host.current ||
        !found_logical || !found_committed) {
      return 13;
    }
  }
  const auto released = device->memory();
  if (released.resident.current != 0u || released.resident.peak != 16u ||
      released.resident.cumulative != 16u || released.host.current != 0u ||
      released.host.peak != 16u || released.host.cumulative != 16u) {
    return 14;
  }

  constexpr std::size_t writer_count = 4u;
  constexpr std::size_t allocation_count = 1'000u;
  auto snapshot_device = rund::compute::open(rund::compute::Target::cpu(2u));
  if (!snapshot_device) {
    return 19;
  }
  std::atomic<std::size_t> finished_writers{};
  std::atomic<bool> allocation_failed{};
  std::array<std::thread, writer_count> writers{};
  for (std::size_t writer = 0u; writer < writers.size(); ++writer) {
    writers[writer] = std::thread{[&, writer] {
      for (std::size_t index = 0u; index < allocation_count; ++index) {
        auto transient = snapshot_device->buffer<std::uint64_t>(
            1u + ((index + writer) % 16u));
        if (!transient) {
          allocation_failed.store(true, std::memory_order_relaxed);
          break;
        }
      }
      finished_writers.fetch_add(1u, std::memory_order_release);
    }};
  }
  bool coherent_snapshots = true;
  while (finished_writers.load(std::memory_order_acquire) != writer_count) {
    const auto snapshot = snapshot_device->memory();
    coherent_snapshots =
        coherent_snapshots && rund_node_memory_contract::ValidStats(snapshot);
  }
  for (auto &writer : writers) {
    writer.join();
  }
  const auto concurrent_released = snapshot_device->memory();
  if (allocation_failed.load(std::memory_order_relaxed) ||
      !coherent_snapshots ||
      !rund_node_memory_contract::ValidStats(concurrent_released) ||
      concurrent_released.resident.current != 0u ||
      concurrent_released.host.current != 0u ||
      concurrent_released.resident.peak == 0u ||
      concurrent_released.host.peak == 0u) {
    return 19;
  }

  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto program = rund::compute::on(rund::compute::Target::cpu(2u))
                     .map<std::int32_t>("twice", input.size(),
                                        [](auto value) { return value * 2; })
                     .compile();
  if (!program) {
    return 3;
  }
  const auto program_memory = program->memory();
  if (!rund_node_memory_contract::ValidStats(program_memory) ||
      program_memory.scope != MemoryScope::Program ||
      program_memory.backend != Backend::Cpu ||
      program_memory.host.current < input.size() * sizeof(std::int32_t) * 2u ||
      program_memory.host.budget != program_memory.host.current) {
    return 4;
  }

  auto job = program->resident(input);
  if (!job) {
    return 5;
  }
  const auto resident = job->memory();
  constexpr std::uint64_t resident_bytes =
      input.size() * sizeof(std::int32_t) * 3u;
  if (!rund_node_memory_contract::ValidStats(resident) ||
      resident.scope != MemoryScope::Job || resident.backend != Backend::Cpu ||
      resident.resident.current != resident_bytes ||
      resident.resident.budget != resident_bytes ||
      resident.host.current < resident_bytes ||
      resident.host.budget != resident.host.current ||
      resident.transfer.current != 0u || resident.transfer.reused != 0u ||
      resident.transfer.budget != 0u ||
      resident.transfer.cumulative != input.size() * sizeof(std::int32_t)) {
    return 6;
  }
  std::array<rund::compute::MemoryEntry, 3u> short_entries{};
  node_compute_allocation::Start();
  const auto short_snapshot = job->memory_snapshot(short_entries);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      short_snapshot.written != short_entries.size() ||
      short_snapshot.total <= short_snapshot.written ||
      !short_snapshot.truncated() ||
      short_snapshot.summary.resident.current != resident_bytes) {
    return 15;
  }
  std::array<rund::compute::MemoryEntry, 32u> entries{};
  const auto snapshot = job->memory_snapshot(entries);
  if (snapshot.truncated() || snapshot.written != snapshot.total ||
      snapshot.written == 0u) {
    return 16;
  }
  bool found_logical_input = false;
  bool found_physical_input = false;
  bool found_traffic = false;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const rund::compute::MemoryEntry &entry = entries[index];
    found_logical_input =
        found_logical_input ||
        (entry.category == rund::compute::MemoryCategory::Resident &&
         entry.use == rund::compute::MemoryUse::Input && entry.index == 0u &&
         entry.bytes.current == input.size() * sizeof(std::int32_t));
    found_physical_input =
        found_physical_input ||
        (entry.category == rund::compute::MemoryCategory::Host &&
         entry.use == rund::compute::MemoryUse::Input && entry.index == 0u &&
         entry.bytes.current == input.size() * sizeof(std::int32_t));
    found_traffic =
        found_traffic ||
        (entry.category == rund::compute::MemoryCategory::Transfer &&
         entry.use == rund::compute::MemoryUse::Traffic &&
         entry.bytes.current == 0u &&
         entry.bytes.peak == resident.transfer.peak &&
         entry.bytes.cumulative == resident.transfer.cumulative &&
         entry.bytes.reused == 0u && entry.bytes.budget == 0u);
  }
  if (!found_logical_input || !found_physical_input || !found_traffic) {
    return 17;
  }

  const auto before_write = job->memory();
  constexpr std::array<std::int32_t, 4u> rewritten{4, 3, 2, 1};
  if (!job->write(rewritten)) {
    return 18;
  }
  const auto after_write = job->memory();
  constexpr std::uint64_t write_bytes = rewritten.size() * sizeof(std::int32_t);
  if (after_write.transfer.current != 0u ||
      after_write.transfer.peak !=
          std::max(before_write.transfer.peak, write_bytes) ||
      after_write.transfer.cumulative !=
          ::rund::detail::counter::SaturatingAdd(
              before_write.transfer.cumulative, write_bytes) ||
      after_write.transfer.reused != 0u || after_write.transfer.budget != 0u ||
      after_write.resident.current != before_write.resident.current ||
      after_write.host.current != before_write.host.current) {
    return 18;
  }

  if (!job->run()) {
    return 7;
  }
  const auto before = job->memory();
  node_compute_allocation::Start();
  const auto warm = job->run();
  const auto after = job->memory();
  node_compute_allocation::Stop();
  if (!warm || node_compute_allocation::Count() != 0u ||
      before.resident.current != after.resident.current ||
      before.tile.current != after.tile.current ||
      (after.tile.current != 0u && after.tile.reused <= before.tile.reused) ||
      before.transfer.cumulative != after.transfer.cumulative) {
    return 8;
  }
  auto output = job->read();
  if (!output) {
    return 9;
  }
  const auto read = job->memory();
  if (read.transfer.current != 0u || read.transfer.reused != 0u ||
      read.transfer.budget != 0u ||
      read.transfer.cumulative !=
          after.transfer.cumulative + input.size() * sizeof(std::int32_t)) {
    return 10;
  }

  const auto backend = device->memory();
  if (!rund_node_memory_contract::ValidStats(backend) ||
      backend.scope != MemoryScope::Backend || backend.host.current != 0u ||
      backend.resident.current != 0u || backend.resident.peak != 16u ||
      backend.resident.cumulative != 16u || backend.transfer.current != 0u ||
      backend.transfer.reused != 0u || backend.transfer.budget != 0u ||
      backend.host.peak != 16u) {
    return 11;
  }
  if (const int owners =
          rund_node_memory_contract::CheckCpuProgramOwnerDeltas();
      owners != 0) {
    return 320 + owners;
  }
  if (const int scratch =
          rund_node_memory_contract::CheckCpuPrimitiveScratchOwnership();
      scratch != 0) {
    return 360 + scratch;
  }
  if (const int collective =
          rund_node_memory_contract::CheckCpuCollectiveScratchOwnership();
      collective != 0) {
    return 380 + collective;
  }
  if (const int graph_storage =
          rund_node_memory_contract::CheckCpuGraphStorageFormula();
      graph_storage != 0) {
    return 300 + graph_storage;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    const int backend_code = 100 * static_cast<int>(backend);
    if (const int memory = rund_node_memory_contract::CheckAccelMemory(backend);
        memory != 0) {
      return 20 + backend_code + memory;
    }
    if (const int owner =
            rund_node_memory_contract::CheckAccelProgramHostAccounting(backend);
        owner != 0) {
      return 40 + backend_code + owner;
    }
    if (const int retained =
            rund_node_memory_contract::CheckRetainedJobMemory(backend);
        retained != 0) {
      return 60 + backend_code + retained;
    }
    if (const int run = rund_node_memory_contract::CheckSortRunMemory(backend);
        run != 0) {
      return 80 + backend_code + run;
    }
  }
  if (rund::node::test_contract::backend_selected(Backend::Metal)) {
    if (const int metal = rund_node_memory_contract::CheckMetalMemoryModel();
        metal != 0) {
      return 290 + metal;
    }
  }
  if (rund::node::test_contract::backend_selected(Backend::Vulkan)) {
    if (const int vulkan = rund_node_memory_contract::CheckVulkanMemoryModel();
        vulkan != 0) {
      return 300 + vulkan;
    }
  }
  return 0;
}
