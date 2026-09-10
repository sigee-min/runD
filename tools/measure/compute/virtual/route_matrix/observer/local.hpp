#pragma once

#include "../internal.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"

namespace rund::measure::compute::route_matrix {

struct RouteObserverImpl final {
  ::rund::compute::detail::DeviceState *device{};
  const ::rund::compute::detail::DeviceOps *original{};
  ::rund::compute::detail::DeviceOps routed{};
  RouteEvidence evidence{};
  bool active{};
  bool owner_seen{};
  std::uintptr_t owner_token{};
  std::uintptr_t control_token{};
  std::uint64_t previous_hash_observations{};
  std::uint64_t previous_hash_reuses{};
  std::uint32_t proof_flags{};
  std::uint32_t capability_flags{};
  std::uint8_t capability_width{};
  std::uint64_t capability_retained{};
  std::uint64_t capability_transient{};
  bool capability_seen{};
  bool proof_identity_seen{};
  std::size_t prepared_count{};
  std::size_t executed_count{};
  bool pending_prepared{};
  bool lifecycle_seen{};
  bool lifecycle_sealed{};

  void
  observe_owner(const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared
                    &) noexcept;
  void
  observe(const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &,
          const ::rund::compute::detail::VirtualExecutionResult &) noexcept;
};

extern thread_local RouteObserverImpl *active_route_observer;

void observe_proof(
    RouteObserverImpl &,
    const ::rund::compute::detail::VirtualDeviceVsmRouteProof &) noexcept;
void mark_lifecycle_failure(RouteObserverImpl &) noexcept;

} // namespace rund::measure::compute::route_matrix
