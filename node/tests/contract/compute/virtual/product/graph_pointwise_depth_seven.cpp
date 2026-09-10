#include "local.hpp"

#include "graph_pointwise_depth_seven/internal.hpp"

#include "../../../target/selection.hpp"
#include "src/compute/virtual/backing.hpp"

namespace rund_node_test_virtual::product {

int CheckProductGraphPointwiseDepthSevenDevice(
    const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  if (backend == Backend::Cpu) {
    auto program = graph_pointwise_depth_seven::build_program(*opened);
    return program && graph_pointwise_depth_seven::validate_program(*program)
               ? 0
               : 2;
  }
  auto prepared = graph_pointwise_depth_seven::prepare_case(*opened);
  if (!prepared.value) {
    return 10 + prepared.reason;
  }
  auto &test_case = *prepared.value;
  const std::uint64_t initial_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  const auto initial_generations =
      graph_pointwise_depth_seven::stage_generations(test_case);
  const Status status = test_case.pipeline.run();
  if (!graph_pointwise_depth_seven::validate_case(
          test_case, backend, status, initial_version, initial_generations)) {
    return 20;
  }
  auto resident_prepared =
      graph_pointwise_depth_seven::prepare_resident_case(*opened);
  if (!resident_prepared.value) {
    return 30 + resident_prepared.reason;
  }
  auto &resident_case = *resident_prepared.value;
  std::array<graph_pointwise_depth_seven::ResidentObservation,
             graph_pointwise_depth_seven::RunCount>
      resident_observations{};
  if (!graph_pointwise_depth_seven::run_resident_case(
          resident_case, resident_observations, *opened, backend)) {
    return 40;
  }
  return graph_pointwise_depth_seven::validate_resident_case(
             resident_case, backend, resident_observations)
             ? 0
             : 50;
}

} // namespace rund_node_test_virtual::product
