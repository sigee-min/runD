#pragma once

#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"

#include <rund/compute.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund_node_test_virtual::product {

enum class RouteDemand : std::uint8_t {
  Natural,
  Required,
  Bypassed,
};

enum class RouteKind : std::uint8_t {
  Unknown,
  CpuRolling,
  AccelRolling,
  Persistent,
  Window,
  DeviceVsm,
  ServiceFreeDirect,
  Rejected,
};

inline constexpr std::uint32_t OwnerAccelRolling = 1u << 0u;
inline constexpr std::uint32_t OwnerPersistent = 1u << 1u;
inline constexpr std::uint32_t OwnerWindow = 1u << 2u;
inline constexpr std::uint32_t OwnerDeviceVsm = 1u << 3u;
inline constexpr std::uint32_t OwnerServiceFreeDirect = 1u << 4u;

struct ProductRouteObservation final {
  static constexpr std::size_t ResidencySubmitReasonCapacity = 128u;
  RouteDemand demand{RouteDemand::Natural};
  RouteKind kind{RouteKind::Unknown};
  bool sliding_prepare_called{};
  bool sliding_execute_called{};
  bool sliding_execute_accepted{};
  bool device_vsm_prepare_called{};
  bool device_vsm_execute_called{};
  bool device_vsm_execute_accepted{};
  bool submit_residency_pipeline_called{};
  bool submit_residency_pipeline_accepted{};
  bool submit_residency_stream_window_called{};
  bool submit_residency_stream_window_accepted{};
  bool submit_residency_schedule_called{};
  bool submit_residency_schedule_accepted{};
  bool prepare_residency_sliding_called{};
  bool submit_residency_sliding_called{};
  bool submit_residency_sliding_accepted{};
  bool window_submit_called{};
  bool window_submit_accepted{};
  std::uint32_t accepted_owner_mask{};
  std::uint32_t accepted_owner_count{};
  std::uint32_t conflict_count{};
  bool completed{};
  rund::compute::Status sliding_prepare_status{
      rund::compute::Status::success()};
  rund::compute::Status sliding_execute_status{
      rund::compute::Status::success()};
  rund::compute::Status prepare_residency_sliding_status{
      rund::compute::Status::success()};
  rund::compute::Status submit_residency_sliding_status{
      rund::compute::Status::success()};
  rund::compute::Status device_vsm_prepare_status{
      rund::compute::Status::success()};
  const char *device_vsm_prepare_reason{};
  bool residency_submit_failed{};
  bool residency_submit_owner_valid{};
  std::uint32_t residency_submit_reason_length{};
  std::array<char, ResidencySubmitReasonCapacity> residency_submit_reason{};
  rund::node::accel::detail::PreparedKernelPipeline residency_submit_owner{};
};

class ProductRouteScope final {
public:
  ProductRouteScope(rund::compute::Device &, ProductRouteObservation &,
                    RouteDemand = RouteDemand::Natural) noexcept;

  ProductRouteScope(const ProductRouteScope &) = delete;
  ProductRouteScope &operator=(const ProductRouteScope &) = delete;

  ~ProductRouteScope();

  [[nodiscard]] explicit operator bool() const noexcept { return active_; }

private:
  std::shared_ptr<rund::compute::detail::DeviceState> device_;
  const rund::compute::detail::DeviceOps *original_{};
  ProductRouteObservation *previous_observation_{};
  const rund::compute::detail::DeviceOps *previous_forward_{};
  rund::compute::detail::DeviceOps routed_{};
  bool active_{};
  bool outer_{};
};

void ResolveProductRoute(ProductRouteObservation &, rund::compute::Backend,
                         bool successful) noexcept;

[[nodiscard]] RouteKind ClassifyMode(rund::compute::Backend,
                                     const rund::compute::ResidencyStats &,
                                     std::uint64_t page_count) noexcept;

// Test-only route selector for contracts that specifically exercise the
// service-aware VSM/Graph Authority. Every production operation remains
// unchanged except the stronger service-free DeviceVsm prepare/execute pair.
class DeviceVsmBypassScope final {
public:
  explicit DeviceVsmBypassScope(rund::compute::Device &) noexcept;

  DeviceVsmBypassScope(const DeviceVsmBypassScope &) = delete;
  DeviceVsmBypassScope &operator=(const DeviceVsmBypassScope &) = delete;

  ~DeviceVsmBypassScope();

  [[nodiscard]] explicit operator bool() const noexcept { return active_; }
  [[nodiscard]] bool device_vsm_available() const noexcept {
    return device_vsm_available_;
  }
  [[nodiscard]] constexpr RouteDemand demand() const noexcept {
    return RouteDemand::Bypassed;
  }

private:
  std::shared_ptr<rund::compute::detail::DeviceState> device_;
  const rund::compute::detail::DeviceOps *original_{};
  ProductRouteObservation *previous_observation_{};
  const rund::compute::detail::DeviceOps *previous_forward_{};
  rund::compute::detail::DeviceOps routed_{};
  bool active_{};
  bool outer_{};
  bool device_vsm_available_{};
};

} // namespace rund_node_test_virtual::product
