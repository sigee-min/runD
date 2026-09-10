#include "local.hpp"

#include "graph_pointwise_depth_six/internal.hpp"

#include "../../../target/selection.hpp"
#include "src/compute/virtual/backing.hpp"

namespace rund_node_test_virtual::product {

int CheckProductGraphPointwiseDepthSixDevice(
    const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  if (backend == Backend::Cpu) {
    auto program = graph_pointwise_depth_six::build_program(*opened);
    return program && graph_pointwise_depth_six::validate_program(*program) ? 0
                                                                            : 2;
  }
  auto prepared = graph_pointwise_depth_six::prepare_case(*opened);
  if (!prepared.value) {
    return 10 + prepared.reason;
  }
  auto &test_case = *prepared.value;
  const std::uint64_t initial_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  const auto initial_generations =
      graph_pointwise_depth_six::stage_generations(test_case);
  const Status status = test_case.pipeline.run();
  return graph_pointwise_depth_six::validate_case(
             test_case, backend, status, initial_version, initial_generations)
             ? 0
             : 20;
}

} // namespace rund_node_test_virtual::product
