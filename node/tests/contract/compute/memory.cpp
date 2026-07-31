#include "memory/local.hpp"

#include "allocation.hpp"
#include "../target/selection.hpp"

#include <array>
#include <cstdint>

int RunComputeMemoryContract() {
  using rund::compute::Backend;
  using rund::compute::MemoryScope;

  if (!rund_node_memory_contract::CheckCounterSaturation()) {
    return 390;
  }
  if (!rund_node_memory_contract::CheckPreparedMemorySnapshot()) {
    return 391;
  }

  if (const int arena = rund_node_memory_contract::CheckValueRouteArena();
      arena != 0) {
    return 400 + arena;
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
    if (active.host.current != 16u || active.host.peak != 16u ||
        active.host.cumulative != 16u) {
      return 13;
    }
  }
  const auto released = device->memory();
  if (released.host.current != 0u || released.host.peak != 16u ||
      released.host.cumulative != 16u) {
    return 14;
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
      resident.transfer.current != 0u ||
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
  if (read.transfer.current != 0u ||
      read.transfer.cumulative !=
          after.transfer.cumulative + input.size() * sizeof(std::int32_t)) {
    return 10;
  }

  const auto backend = device->memory();
  if (!rund_node_memory_contract::ValidStats(backend) ||
      backend.scope != MemoryScope::Backend || backend.host.current != 0u ||
      backend.transfer.current != 0u || backend.host.peak != 16u) {
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
    if (const int memory =
            rund_node_memory_contract::CheckAccelMemory(backend);
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
    if (const int run =
            rund_node_memory_contract::CheckSortRunMemory(backend);
        run != 0) {
      return 80 + backend_code + run;
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
