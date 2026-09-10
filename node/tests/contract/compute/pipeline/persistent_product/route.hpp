#pragma once

#include "src/accel/kernel/residency/persistent_sliding.hpp"

#include <rund/compute/status.hpp>

#include <memory>

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund_node_test_persistent_product {

class PersistentProductBacking;

struct ProductRouteObservation final {
  std::shared_ptr<void> owner{};
  rund::node::accel::detail::PersistentResidencySlidingFinal final{};
  rund::compute::Status prepare_status{
      rund::compute::Status::fail(rund::compute::Reason::BackendUnsupported)};
  PersistentProductBacking *fail_input{};
  rund::node::accel::detail::PersistentResidencySlidingMode prepared_mode{
      rund::node::accel::detail::PersistentResidencySlidingMode::OneSubmit};
  rund::node::accel::detail::PersistentResidencySlidingMode request_mode{
      rund::node::accel::detail::PersistentResidencySlidingMode::OneSubmit};
  rund::node::accel::detail::PersistentResidencySlidingMode control_mode{
      rund::node::accel::detail::PersistentResidencySlidingMode::OneSubmit};
  rund::node::accel::detail::PersistentResidencySlidingMode capability_mode{
      rund::node::accel::detail::PersistentResidencySlidingMode::OneSubmit};
  bool mode_snapshot{};
  bool production_route{};
  bool prepared{};
  bool final_received{};
  // Test-only route substitution used to prove that a terminal
  // BackendUnsupported result cannot be reinterpreted as rolling fallback.
  bool force_terminal_unsupported{};
};

[[nodiscard]] rund::compute::Status RunThroughPersistentProductRoute(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &,
    ProductRouteObservation &) noexcept;

} // namespace rund_node_test_persistent_product
