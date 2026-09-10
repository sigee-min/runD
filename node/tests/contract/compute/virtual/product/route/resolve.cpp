#include "../model.hpp"
#include "../route.hpp"

namespace rund_node_test_virtual::product {

RouteKind ClassifyMode(const rund::compute::Backend backend,
                       const rund::compute::ResidencyStats &residency,
                       const std::uint64_t page_count) noexcept {
  if (backend == rund::compute::Backend::Cpu) {
    return RouteKind::CpuRolling;
  }
  if ((backend != rund::compute::Backend::Metal &&
       backend != rund::compute::Backend::Vulkan) ||
      page_count == 0u) {
    return RouteKind::Rejected;
  }
  if (residency.window_handoff_count == 1u &&
      residency.window_batch_count == 1u &&
      residency.window_queue_call_count == 1u) {
    return RouteKind::DeviceVsm;
  }
  const std::uint64_t epochs =
      page_count / FrameCapacity +
      static_cast<std::uint64_t>(page_count % FrameCapacity != 0u);
  if (residency.window_handoff_count == 1u &&
      residency.window_batch_count == epochs &&
      residency.window_queue_call_count == 1u) {
    return RouteKind::Persistent;
  }
  return RouteKind::AccelRolling;
}

void ResolveProductRoute(ProductRouteObservation &observation,
                         const rund::compute::Backend backend,
                         const bool successful) noexcept {
  observation.completed = successful;
  if (!successful || observation.conflict_count != 0u) {
    observation.kind = RouteKind::Rejected;
    return;
  }
  const std::uint32_t owners = observation.accepted_owner_mask;
  constexpr std::uint32_t recognized = OwnerAccelRolling | OwnerPersistent |
                                       OwnerWindow | OwnerDeviceVsm |
                                       OwnerServiceFreeDirect;
  if (owners == 0u) {
    if (backend == rund::compute::Backend::Cpu &&
        observation.demand == RouteDemand::Natural &&
        observation.accepted_owner_count == 0u) {
      observation.kind = RouteKind::CpuRolling;
    } else {
      observation.kind = RouteKind::Rejected;
    }
    return;
  }
  if (observation.accepted_owner_count == 0u || (owners & ~recognized) != 0u ||
      (owners & (owners - 1u)) != 0u) {
    observation.kind = RouteKind::Rejected;
    return;
  }
  if (observation.demand == RouteDemand::Required &&
      (owners & OwnerDeviceVsm) == 0u) {
    observation.kind = RouteKind::Rejected;
    return;
  }
  if ((owners & OwnerDeviceVsm) != 0u &&
      (owners & (OwnerPersistent | OwnerWindow | OwnerServiceFreeDirect)) ==
          0u) {
    observation.kind = RouteKind::DeviceVsm;
    return;
  }
  if ((owners & OwnerPersistent) != 0u &&
      (owners & (OwnerWindow | OwnerDeviceVsm | OwnerServiceFreeDirect)) ==
          0u) {
    observation.kind = RouteKind::Persistent;
    return;
  }
  // Window has no exact public terminal Stats oracle in this aggregate test;
  // accepted submission therefore remains fail-closed.
  if ((owners & OwnerWindow) != 0u) {
    observation.kind = RouteKind::Rejected;
    return;
  }
  if ((owners & OwnerServiceFreeDirect) != 0u) {
    observation.kind = RouteKind::ServiceFreeDirect;
    return;
  }
  if ((owners & OwnerAccelRolling) != 0u) {
    observation.kind = RouteKind::AccelRolling;
    return;
  }
  observation.kind = RouteKind::Rejected;
}

} // namespace rund_node_test_virtual::product
