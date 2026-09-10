#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

bool initialize_persistent_service_coordinate(
    SlidingProductRun &state, const std::uint64_t coordinate,
    PersistentServiceCoordinate &service) noexcept {
  service = {};
  service.coordinate = coordinate;
  service.slot = static_cast<std::size_t>(coordinate % state.cold->role_count);
  service.turn = coordinate / state.cold->role_count;
  service.work = &state.work[service.slot];
  if (!node::accel::detail::persistent_sliding_service_identity(
          state.persistent_preparation.active_request(), coordinate,
          service.identity)) {
    record_service_fault(state, ServiceFaultStage::Init, 1u, coordinate,
                         Status::fail(Reason::CompletionInvalid), 1u);
    return false;
  }
  if (!reset_work(*service.work, coordinate)) {
    record_service_fault(state, ServiceFaultStage::Init, 2u, coordinate,
                         Status::fail(Reason::CompletionInvalid), 2u);
    return false;
  }
  if (!ensure_projection(state, *service.work, coordinate)) {
    record_service_fault(state, ServiceFaultStage::Init, 3u, coordinate,
                         Status::fail(Reason::CompletionInvalid), 3u);
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
