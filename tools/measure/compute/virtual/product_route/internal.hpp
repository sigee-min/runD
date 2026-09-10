#pragma once

#include "../product_route.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <algorithm>
#include <limits>

namespace rund::measure::compute::virtual_residency {

struct ProductRouteObserverImpl final {
  ::rund::compute::detail::DeviceState *device{};
  const ::rund::compute::detail::DeviceOps *original{};
  ::rund::compute::detail::DeviceOps routed{};
  ProductRouteEvidence evidence{};
  bool active{};
  bool owner_seen{};
  std::uintptr_t owner_token{};
  std::uintptr_t control_token{};
  std::uint64_t previous_hash_observations{};
  std::uint64_t previous_hash_reuses{};

  void
  observe_owner(const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared
                    &) noexcept;
  void
  observe(const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &,
          const ::rund::compute::detail::VirtualExecutionResult &) noexcept;
};

extern thread_local ProductRouteObserverImpl *active_product_route_observer;

[[nodiscard]] ::rund::compute::Status prepare_product_route_observed(
    ::rund::compute::detail::VirtualPipelineState &,
    const ::rund::compute::detail::VirtualRunProjection &,
    ::rund::compute::detail::VirtualDeviceVsmRouteProof,
    ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &) noexcept;
[[nodiscard]] ::rund::compute::detail::VirtualExecutionResult
execute_product_route_observed(
    ::rund::compute::detail::VirtualPipelineState &,
    std::span<::rund::compute::VirtualBacking *const>,
    ::rund::compute::VirtualBacking &,
    const ::rund::compute::detail::VirtualRunProjection &,
    const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &,
    ::rund::compute::Stats &) noexcept;

inline void add_saturated(std::uint64_t &target,
                          const std::uint64_t value) noexcept {
  target = value > std::numeric_limits<std::uint64_t>::max() - target
               ? std::numeric_limits<std::uint64_t>::max()
               : target + value;
}

[[nodiscard]] bool exact_zero(const ProductRouteEvidence &) noexcept;

} // namespace rund::measure::compute::virtual_residency
