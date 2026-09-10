#pragma once

#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <rund/compute/status.hpp>

#include <memory>

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund_node_test_device_vsm_product {

struct RouteObservation final {
  std::shared_ptr<void> owner{};
  rund::compute::detail::device_vsm_product_detail::DeviceVsmProductEvidence
      evidence{};
  bool production_route{};
  bool prepared{};
  bool executed{};
  bool execution_status{};
  bool execution_poison{};
  unsigned execution_reason{};
  const char *prepare_reason{};
};

[[nodiscard]] rund::compute::Status RunThroughDeviceVsmProductRoute(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &,
    RouteObservation &) noexcept;

[[nodiscard]] rund::compute::Status RunThroughDeviceVsmProductRoute(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &,
    RouteObservation &, bool force_device_vsm) noexcept;

} // namespace rund_node_test_device_vsm_product
