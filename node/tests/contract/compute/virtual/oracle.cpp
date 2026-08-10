#include "local.hpp"

#include "../allocation.hpp"
#include "backing.hpp"
#include "golden.hpp"
#include "model.hpp"
#include "pager.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund_node_test_virtual {

int CheckVirtualBackingOracle() {
  std::array<std::int32_t, LogicalElements> seed{};
  std::array<std::int32_t, LogicalElements> observed{};
  SeedInput(seed);

  MemoryVirtualBacking input{BackingBytes};
  MemoryVirtualBacking output{BackingBytes};
  if (!input.seed(0u, std::as_bytes(std::span{seed})) ||
      input.size_bytes() != BackingBytes ||
      output.size_bytes() != BackingBytes ||
      !input.tail_is(LogicalBytes, TailPoison) ||
      !output.tail_is(LogicalBytes, TailPoison)) {
    return 1;
  }

  const void *const input_identity = input.identity();
  const void *const output_identity = output.identity();
  MemoryVirtualPager pager{};
  if (!pager.run(input, output, LogicalElements) ||
      output.facts().observation_count != 0u) {
    return 2;
  }

  node_compute_allocation::Start();
  bool warm_ok = true;
  for (std::size_t run = 0u; run < WarmRunCount; ++run) {
    warm_ok = warm_ok && pager.run(input, output, LogicalElements);
  }
  const BackingFacts before_observation = output.facts();
  const bool observed_ok = output.observe(
      0u, std::as_writable_bytes(std::span<std::int32_t>{observed}));
  node_compute_allocation::Stop();
  if (!warm_ok || !observed_ok || node_compute_allocation::Count() != 0u ||
      before_observation.observation_count != 0u) {
    return 3;
  }

  const PagerFacts pager_facts = pager.facts();
  const BackingFacts input_facts = input.facts();
  const BackingFacts output_facts = output.facts();
  constexpr std::uint64_t total_runs = WarmRunCount + 1u;
  constexpr std::uint64_t total_epochs = total_runs * PageCount;
  if (pager_facts.run_count != total_runs ||
      pager_facts.epoch_count != total_epochs ||
      pager_facts.resident_capacity_bytes != ResidentCapacityBytes ||
      pager_facts.resident_current_bytes != 0u ||
      pager_facts.resident_peak_bytes != ResidentCapacityBytes ||
      input_facts.load_count != total_epochs ||
      input_facts.load_bytes != total_runs * LogicalBytes ||
      input_facts.store_count != 0u || input_facts.observation_count != 0u ||
      output_facts.load_count != 0u ||
      output_facts.store_count != total_epochs ||
      output_facts.store_bytes != total_runs * LogicalBytes ||
      output_facts.observation_count != 1u ||
      output_facts.observation_bytes != LogicalBytes) {
    return 4;
  }

  if (input.identity() != input_identity ||
      output.identity() != output_identity ||
      input.size_bytes() != BackingBytes ||
      output.size_bytes() != BackingBytes ||
      !input.tail_is(LogicalBytes, TailPoison) ||
      !output.tail_is(LogicalBytes, TailPoison)) {
    return 5;
  }
  if (!GoldenMatches(observed) || HashValues(observed) != GoldenHash) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_virtual
