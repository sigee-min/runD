#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <span>

namespace rund_node_test_virtual::product::graph_resident_host {

bool capture_run(Case &test_case, Observation &observation) noexcept {
  observation.after = test_case.pipeline.stats();
  observation.output_hash = observation.after.output_hash;
  observation.version_after =
      rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
  observation.output.resize(ElementCount);
  observation.output_match = test_case.output->observe(std::as_writable_bytes(
                                 std::span{observation.output})) &&
                             observation.output == test_case.expected;
  for (std::size_t input = 0u; input < InputCount; ++input) {
    observation.inputs[input] = test_case.inputs[input]->facts();
  }
  observation.output_facts = test_case.output->facts();
  if (test_case.state == nullptr || test_case.state->pipeline == nullptr) {
    return false;
  }
  observation.receipts_idle = test_case.state->cpu_receipts != nullptr &&
                              test_case.state->cpu_receipts->idle();
  observation.quarantine_empty =
      test_case.state->cpu_quarantine_hold != nullptr &&
      test_case.state->cpu_quarantine_hold->book ==
          test_case.state->cpu_receipts &&
      test_case.state->cpu_quarantine_hold->empty();
  observation.failure = test_case.state->failure_log;
  observation.distinct_backings = true;
  for (std::size_t left = 0u; left < InputCount; ++left) {
    for (std::size_t right = left + 1u; right < InputCount; ++right) {
      observation.distinct_backings = observation.distinct_backings &&
                                      test_case.inputs[left]->identity() !=
                                          test_case.inputs[right]->identity();
    }
    observation.distinct_backings =
        observation.distinct_backings &&
        test_case.inputs[left]->identity() != test_case.output->identity();
  }
  return true;
}

} // namespace rund_node_test_virtual::product::graph_resident_host
