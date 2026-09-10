#include "local.hpp"

#include "graph_wavefront_host/internal.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"
#include "src/compute/virtual/backing.hpp"

namespace rund_node_test_virtual::product {

int CheckProductGraphHostWavefront(const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  auto prepared = graph_wavefront_host::prepare_case(*opened);
  if (!prepared.value) {
    return prepared.reason;
  }
  auto &test_case = *prepared.value;
  if (backend == Backend::Vulkan &&
      !graph_wavefront_host::check_graph_admission(*test_case.state)) {
    return 6;
  }
  ProductRouteObservation route_observation{};
  ProductRouteScope route_scope{*opened, route_observation};
  if (!route_scope) {
    return 6;
  }
  const std::uint64_t initial_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  const Status first_run = test_case.pipeline.run();
  const bool valid = graph_wavefront_host::validate_case(
      test_case, backend, first_run, initial_version, route_observation);
  return valid ? 0 : 7;
}

} // namespace rund_node_test_virtual::product
