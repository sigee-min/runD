#include "local.hpp"

#include "graph_pointwise_wide_host/internal.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"
#include "src/compute/virtual/backing.hpp"

namespace rund_node_test_virtual::product {

int CheckProductGraphPointwiseWideHost(const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  DeviceVsmBypassScope route{*opened};
  if (!route || !route.device_vsm_available()) {
    return 2;
  }
  auto prepared = graph_pointwise_wide_host::prepare_case(*opened);
  if (!prepared.value) {
    return 10 + prepared.reason;
  }
  auto &test_case = *prepared.value;
  const std::uint64_t initial_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  ProductRouteObservation observation{};
  Status status = Status::fail(Reason::PipelineInvalid);
  {
    ProductRouteScope route_scope{*opened, observation, RouteDemand::Bypassed};
    if (!route_scope) {
      return 2;
    }
    status = test_case.pipeline.run();
  }
  ResolveProductRoute(observation, backend, static_cast<bool>(status));
  const bool valid = graph_pointwise_wide_host::validate_case(
      test_case, backend, status, initial_version);
  if (!valid) {
    graph_pointwise_wide_host::report_route_evidence(test_case, backend,
                                                     observation);
  }
  return valid ? 0 : 20;
}

} // namespace rund_node_test_virtual::product
