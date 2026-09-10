#include "local.hpp"

#include "graph_pointwise_wide_host/internal.hpp"

#include "../../../target/selection.hpp"
#include "src/compute/virtual/backing.hpp"

namespace rund_node_test_virtual::product {

int CheckProductGraphPointwiseWideDevice(const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  auto prepared = graph_pointwise_wide_host::prepare_case(*opened);
  if (!prepared.value) {
    return 10 + prepared.reason;
  }
  auto &test_case = *prepared.value;
  const std::uint64_t initial_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  const auto initial_generations =
      graph_pointwise_wide_host::device_stage_generations(test_case);
  const Status status = test_case.pipeline.run();
  return graph_pointwise_wide_host::validate_device_case(
             test_case, backend, status, initial_version, initial_generations)
             ? 0
             : 20;
}

} // namespace rund_node_test_virtual::product
