#include "../../local.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../../../src/compute/cpu/graph.hpp"
#include "../../../../../../src/compute/job/state.hpp"

#include <rund/compute.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_memory_contract {

int CheckCpuCollectiveScratchOwnership() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t count = 1024u;
  const std::vector<std::uint32_t> input(count, 1u);
  auto device = open(Target::cpu(2u));
  if (!device) {
    return 1;
  }
  auto scan_program =
      on(*device)
          .input<std::uint32_t>(count)
          .map("collective-scan-input", [](auto value) { return value; })
          .scan(Scan::InclusiveSum)
          .compile();
  auto reduce_program =
      on(*device)
          .input<std::uint32_t>(count)
          .map("collective-reduce-input", [](auto value) { return value; })
          .reduce(Reduce::Sum)
          .compile();
  if (!scan_program || !reduce_program) {
    return 2;
  }
  auto scan_job = scan_program->resident(input);
  auto reduce_job = reduce_program->resident(input);
  if (!scan_job || !reduce_job) {
    return 3;
  }
  const auto scan_state = JobAccess::state(*scan_job);
  const auto reduce_state = JobAccess::state(*reduce_job);
  const auto active_collective = [](const std::shared_ptr<JobState> &state) {
    if (state == nullptr || state->cpu == nullptr ||
        state->cpu->graph == nullptr || state->cpu->graph->storage == nullptr) {
      return static_cast<CpuCollectiveRun *>(nullptr);
    }
    for (auto &collective : state->cpu->graph->storage->collectives) {
      return &collective;
    }
    return static_cast<CpuCollectiveRun *>(nullptr);
  };
  const CpuCollectiveRun *const scan = active_collective(scan_state);
  const CpuCollectiveRun *const reduce = active_collective(reduce_state);
  if (scan == nullptr || reduce == nullptr || !scan->needs_prefixes ||
      reduce->needs_prefixes || scan->totals.empty() ||
      scan->prefixes.size() != scan->totals.size() || reduce->totals.empty() ||
      !reduce->prefixes.empty() || !reduce->prefix_capacity.empty()) {
    return 4;
  }
  if (!scan_job->run() || !reduce_job->run()) {
    return 5;
  }
  return 0;
}

} // namespace rund_node_memory_contract
