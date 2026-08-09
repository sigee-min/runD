#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {

[[nodiscard]] int CheckNestedWindow(rund::compute::Device &device,
                                    const rund::compute::Backend backend) {
  auto seed = MakeNestedSeedProgram(device);
  auto action = MakeNestedActionProgram(device);
  auto fold = MakeNestedFoldProgram(device);
  if (!seed || !action || !fold) {
    return 1;
  }
  for (const std::uint32_t count : kCounts) {
    const int checked =
        CheckNestedCount(device, backend, *seed, *action, *fold, count);
    if (checked != 0) {
      return 10 + checked;
    }
  }
  const int binding_identity =
      CheckTransactionalBindingIdentity(device, backend);
  if (binding_identity != 0) {
    return 20 + binding_identity;
  }
  const int terminal = CheckNestedTerminal(device, backend);
  if (terminal != 0) {
    return 30 + terminal;
  }
  const int failures = CheckNestedFailures(device, backend);
  if (failures != 0) {
    return 80 + failures;
  }
  const int profile =
      CheckNestedProfileCase(device, backend, *seed, *action, *fold);
  if (profile != 0) {
    return 150 + profile;
  }
  const int reuse = CheckNestedRetainedReuse(device, *seed, *fold);
  if (reuse != 0) {
    return 160 + reuse;
  }
  const int composition =
      CheckNestedComposition(device, backend, *seed, *action, *fold);
  if (composition != 0) {
    return 170 + composition;
  }
  const int aggregate = CheckNestedAggregateStats(device, backend);
  if (aggregate != 0) {
    return 175 + aggregate;
  }
  const int dormant_aggregate = CheckDormantAggregateRoutes(device, backend);
  if (dormant_aggregate != 0) {
    return 176 + dormant_aggregate;
  }
  const int workspace_observation = CheckWorkspaceObservationCapacity(device);
  if (workspace_observation != 0) {
    return 177 + workspace_observation;
  }
  const int aggregate_failures =
      CheckNestedAggregateSeedFailures(device, backend, *seed, *action, *fold);
  if (aggregate_failures != 0) {
    return 178 + aggregate_failures;
  }
  const int maximum = CheckNestedMaximumPlan(device, *action, *fold);
  if (maximum != 0) {
    return 180 + maximum;
  }
  const int product = CheckProductPlan(device, backend);
  return product == 0 ? 0 : 190 + product;
}

} // namespace rund::node::test_contract::window
