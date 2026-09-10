#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <limits>

namespace rund_node_test_virtual::product::graph_pointwise {

int check_scale(const rund::compute::Device &device,
                const rund::compute::Backend backend,
                const std::size_t page_count) {
  Preparation prepared = prepare_case(device, page_count);
  if (!prepared.value) {
    return prepared.reason;
  }
  Case &test_case = *prepared.value;
  const std::uint64_t version =
      rund::compute::detail::VirtualBackingAccess::version(
          *test_case.output_backing);
  const auto generations = stage_generations(test_case);
  const rund::compute::Status status = test_case.pipeline.run();
  const auto published = stage_generations(test_case);
  bool stages_published = backend == rund::compute::Backend::Cpu;
  if (backend != rund::compute::Backend::Cpu) {
    stages_published = true;
    for (std::size_t stage = 0u; stage < StageCount; ++stage) {
      stages_published =
          stages_published &&
          generations[stage] != std::numeric_limits<std::uint64_t>::max() &&
          published[stage] == generations[stage] + 1u;
    }
  }
  return validate_success(test_case, backend, status, version) &&
                 stages_published
             ? 0
             : 5;
}

} // namespace rund_node_test_virtual::product::graph_pointwise
