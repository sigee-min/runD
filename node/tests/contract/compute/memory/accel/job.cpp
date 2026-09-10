#include "../model.hpp"

#include "../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../../src/compute/job/state.hpp"

#include <array>
#include <memory>

namespace rund_node_memory_contract {

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
  if (resident.transfer.current != 0u || resident.transfer.reused != 0u ||
      resident.transfer.budget != 0u ||
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
  const std::shared_ptr<detail::BufferState> input_owner =
      state->inputs.empty() ? nullptr : state->inputs.front();
  bool found_staging = false;
  bool found_logical_input = false;
  bool found_physical_input = false;
  bool found_traffic = false;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const MemoryEntry &entry = entries[index];
    found_staging =
        found_staging || (entry.category == MemoryCategory::Staging &&
                          entry.bytes.current == prepared.current);
    found_logical_input =
        found_logical_input ||
        (input_owner != nullptr && entry.category == MemoryCategory::Resident &&
         entry.use == MemoryUse::Input && entry.index == 0u &&
         entry.bytes.current == input_owner->bytes);
    found_physical_input =
        found_physical_input ||
        (input_owner != nullptr && entry.category == MemoryCategory::Device &&
         entry.use == MemoryUse::Input && entry.index == 0u &&
         entry.bytes.current == input_owner->physical_bytes);
    found_traffic =
        found_traffic ||
        (entry.category == MemoryCategory::Transfer &&
         entry.use == MemoryUse::Traffic && entry.bytes.current == 0u &&
         entry.bytes.peak == resident.transfer.peak &&
         entry.bytes.cumulative == resident.transfer.cumulative &&
         entry.bytes.reused == 0u && entry.bytes.budget == 0u);
  }
  if (!found_staging || !found_logical_input || !found_physical_input ||
      !found_traffic) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_memory_contract
