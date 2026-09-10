#include "internal.hpp"

namespace rund_node_test_virtual::product {

ProductRouteScope::ProductRouteScope(rund::compute::Device &device,
                                     ProductRouteObservation &observation,
                                     const RouteDemand demand) noexcept
    : device_(rund::compute::detail::DeviceAccess::state(device)) {
  observation = {};
  observation.demand = demand;
  if (device_ == nullptr) {
    return;
  }
  std::unique_lock lock{route_detail::protocol_mutex};
  if (!route_detail::enter_protocol(lock, *device_, &observation, original_,
                                    previous_observation_, previous_forward_,
                                    outer_)) {
    return;
  }
  if (original_ == nullptr) {
    route_detail::protocol.forward = original_;
    route_detail::protocol.observation = &observation;
    ++route_detail::protocol.depth;
    ++route_detail::protocol.inflight;
    active_ = true;
    return;
  }
  routed_ = *original_;
  if (original_->virtual_execution.prepare_virtual_sliding_product != nullptr &&
      original_->virtual_execution.execute_virtual_sliding_product != nullptr) {
    routed_.virtual_execution.prepare_virtual_sliding_product =
        route_detail::ObserveSlidingPrepare;
    routed_.virtual_execution.execute_virtual_sliding_product =
        route_detail::ObserveSlidingExecute;
  }
  if (original_->virtual_execution.prepare_virtual_device_vsm_product !=
          nullptr &&
      original_->virtual_execution.execute_virtual_device_vsm_product !=
          nullptr) {
    routed_.virtual_execution.prepare_virtual_device_vsm_product =
        route_detail::ObserveDeviceVsmPrepare;
    routed_.virtual_execution.execute_virtual_device_vsm_product =
        route_detail::ObserveDeviceVsmExecute;
  }
  if (original_->virtual_execution.run_service_free_direct_product != nullptr) {
    routed_.virtual_execution.run_service_free_direct_product =
        route_detail::ObserveServiceFreeDirect;
  }
  if (original_->residency.submit_residency_pipeline != nullptr) {
    routed_.residency.submit_residency_pipeline =
        route_detail::ObserveResidencyPipeline;
  }
  if (original_->residency.submit_residency_window != nullptr) {
    routed_.residency.submit_residency_window =
        route_detail::ObserveResidencyWindow;
  }
  if (original_->residency.submit_residency_stream_window != nullptr) {
    routed_.residency.submit_residency_stream_window =
        route_detail::ObserveResidencyStreamWindow;
  }
  if (original_->residency.submit_residency_schedule != nullptr) {
    routed_.residency.submit_residency_schedule =
        route_detail::ObserveResidencySchedule;
  }
  if (original_->residency.prepare_residency_sliding != nullptr) {
    routed_.residency.prepare_residency_sliding =
        route_detail::ObservePrepareResidencySliding;
  }
  if (original_->residency.submit_residency_sliding != nullptr) {
    routed_.residency.submit_residency_sliding =
        route_detail::ObserveSubmitResidencySliding;
  }
  route_detail::protocol.forward = original_;
  route_detail::protocol.observation = &observation;
  ++route_detail::protocol.depth;
  ++route_detail::protocol.inflight;
  device_->ops = &routed_;
  active_ = true;
}

ProductRouteScope::~ProductRouteScope() {
  if (active_ && device_ != nullptr) {
    route_detail::finish_protocol(*device_, original_, previous_observation_,
                                  previous_forward_, &routed_, outer_);
  }
}

DeviceVsmBypassScope::DeviceVsmBypassScope(
    rund::compute::Device &device) noexcept
    : device_(rund::compute::detail::DeviceAccess::state(device)) {
  if (device_ == nullptr) {
    return;
  }
  std::unique_lock lock{route_detail::protocol_mutex};
  if (!route_detail::enter_protocol(lock, *device_, nullptr, original_,
                                    previous_observation_, previous_forward_,
                                    outer_)) {
    return;
  }
  if (original_ == nullptr) {
    ++route_detail::protocol.depth;
    ++route_detail::protocol.inflight;
    active_ = true;
    return;
  }
  const bool prepare =
      original_->virtual_execution.prepare_virtual_device_vsm_product !=
      nullptr;
  const bool execute =
      original_->virtual_execution.execute_virtual_device_vsm_product !=
      nullptr;
  if (prepare != execute) {
    if (outer_) {
      route_detail::protocol = {};
      route_detail::protocol_cv.notify_all();
    }
    return;
  }
  device_vsm_available_ = prepare;
  routed_ = *original_;
  routed_.virtual_execution.prepare_virtual_device_vsm_product = nullptr;
  routed_.virtual_execution.execute_virtual_device_vsm_product = nullptr;
  ++route_detail::protocol.depth;
  ++route_detail::protocol.inflight;
  device_->ops = &routed_;
  active_ = true;
}

DeviceVsmBypassScope::~DeviceVsmBypassScope() {
  if (active_ && device_ != nullptr) {
    route_detail::finish_protocol(*device_, original_, previous_observation_,
                                  previous_forward_, &routed_, outer_);
  }
}

} // namespace rund_node_test_virtual::product
