#include "product_route/internal.hpp"

namespace rund::measure::compute::virtual_residency {

thread_local ProductRouteObserverImpl *active_product_route_observer{};

ProductRouteObserver::ProductRouteObserver(
    ::rund::compute::Device &device) noexcept {
  try {
    impl_ = std::make_unique<ProductRouteObserverImpl>();
  } catch (...) {
    return;
  }
  const auto &state = ::rund::compute::detail::DeviceAccess::state(device);
  if (state == nullptr || state->ops == nullptr ||
      active_product_route_observer != nullptr) {
    return;
  }
  impl_->device = state.get();
  impl_->original = state->ops;
  const bool prepare =
      impl_->original->virtual_execution.prepare_virtual_device_vsm_product !=
      nullptr;
  const bool execute =
      impl_->original->virtual_execution.execute_virtual_device_vsm_product !=
      nullptr;
  impl_->evidence.production_slots = prepare && execute;
  if (!impl_->evidence.production_slots) {
    return;
  }
  impl_->routed = *impl_->original;
  impl_->routed.virtual_execution.prepare_virtual_device_vsm_product =
      prepare_product_route_observed;
  impl_->routed.virtual_execution.execute_virtual_device_vsm_product =
      execute_product_route_observed;
  impl_->device->ops = &impl_->routed;
  active_product_route_observer = impl_.get();
  impl_->active = true;
}

ProductRouteObserver::~ProductRouteObserver() {
  if (impl_ == nullptr || !impl_->active) {
    return;
  }
  impl_->device->ops = impl_->original;
  active_product_route_observer = nullptr;
}

void ProductRouteObserver::reset() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  const bool production_slots = impl_->evidence.production_slots;
  impl_->evidence = {};
  impl_->evidence.production_slots = production_slots;
}

const ProductRouteEvidence &ProductRouteObserver::evidence() const noexcept {
  static const ProductRouteEvidence unavailable{};
  return impl_ == nullptr ? unavailable : impl_->evidence;
}

} // namespace rund::measure::compute::virtual_residency
