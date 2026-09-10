#include "local.hpp"

#include "src/compute/virtual/backing.hpp"

#include <array>

namespace rund_node_test_virtual::product::graph_resident::fixture_detail {

bool run_once(Case &test_case, Observation &observation,
              rund::compute::Device &device,
              const rund::compute::Backend backend) {
  ProductRouteObservation route{};
  {
    ProductRouteScope route_scope{device, route};
    if (!route_scope) {
      return false;
    }
    observation.before = test_case.pipeline.stats();
    observation.version_before =
        rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
    observation.staged_output =
        test_case.output != nullptr &&
        rund::compute::detail::VirtualBackingAccess::resident(
            *test_case.output) == nullptr;
    observation.status = test_case.pipeline.run();
    observation.device_vsm_prepare_reason = route.device_vsm_prepare_reason;
    if (!capture_run(test_case, observation)) {
      return false;
    }
  }
  ResolveProductRoute(route, backend, static_cast<bool>(observation.status));
  observation.route_kind = route.kind;
  observation.accepted_owner_mask = route.accepted_owner_mask;
  observation.accepted_owner_count = route.accepted_owner_count;
  return true;
}

bool run_u32_once(U32Case &test_case, Observation &observation,
                  rund::compute::Device &device,
                  const rund::compute::Backend backend) {
  ProductRouteObservation route{};
  {
    ProductRouteScope route_scope{device, route};
    if (!route_scope) {
      return false;
    }
    observation.before = test_case.pipeline.stats();
    observation.version_before =
        rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
    observation.staged_output =
        test_case.output != nullptr &&
        rund::compute::detail::VirtualBackingAccess::resident(
            *test_case.output) == nullptr;
    observation.status = test_case.pipeline.run();
    observation.device_vsm_prepare_reason = route.device_vsm_prepare_reason;
    if (!capture_u32_run(test_case, observation)) {
      return false;
    }
  }
  ResolveProductRoute(route, backend, static_cast<bool>(observation.status));
  observation.route_kind = route.kind;
  observation.accepted_owner_mask = route.accepted_owner_mask;
  observation.accepted_owner_count = route.accepted_owner_count;
  return true;
}

namespace {

[[nodiscard]] bool run_u32_probe(const rund::compute::Backend backend,
                                 rund::compute::Device &device) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  (void)backend;
  (void)device;
  return true;
#else
  auto prepared = prepare_u32_case(device);
  if (!prepared.value) {
    std::fprintf(stderr, "GraphResident U32 prepare reason=%d\n",
                 prepared.reason);
    return false;
  }
  U32Case &test_case = *prepared.value;
  std::array<Observation, RunCount> observations{};
  if (!run_u32_case(test_case, observations, device, backend)) {
    return false;
  }
  return validate_u32_case(test_case, backend, observations);
#endif
}

} // namespace

} // namespace rund_node_test_virtual::product::graph_resident::fixture_detail

namespace rund_node_test_virtual::product::graph_resident {

using fixture_detail::run_once;
using fixture_detail::run_staged_probe;
using fixture_detail::run_u32_once;
using fixture_detail::run_u32_probe;

bool run_case(Case &test_case, std::array<Observation, RunCount> &observations,
              rund::compute::Device &device,
              const rund::compute::Backend backend) {
  for (std::size_t index = 0u; index < RunCount; ++index) {
    Observation &observation = observations[index];
    if (!run_once(test_case, observation, device, backend)) {
      return false;
    }
  }
  const bool resident = !test_case.inputs.empty() &&
                        test_case.inputs[0u] != nullptr &&
                        rund::compute::detail::VirtualBackingAccess::resident(
                            *test_case.inputs[0u]) != nullptr;
  if (!resident || !run_staged_probe(backend, test_case.variant)) {
    return !resident;
  }
  return test_case.variant != Variant::Ordinary ||
         run_u32_probe(backend, device);
}

bool run_u32_case(U32Case &test_case,
                  std::array<Observation, RunCount> &observations,
                  rund::compute::Device &device,
                  const rund::compute::Backend backend) {
  for (Observation &observation : observations) {
    if (!run_u32_once(test_case, observation, device, backend)) {
      return false;
    }
  }
  return !test_case.inputs.empty() && test_case.inputs[0u] != nullptr &&
         rund::compute::detail::VirtualBackingAccess::resident(
             *test_case.inputs[0u]) != nullptr;
}

} // namespace rund_node_test_virtual::product::graph_resident
