#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

bool select_persistent_service_native(
    SlidingProductRun &state, PersistentServiceCoordinate &service) noexcept {
  node::accel::detail::PreparedResidencySlidingSelection selected{};
  return select_native(state, *service.work, service.turn,
                       static_cast<std::uint8_t>(service.slot),
                       selected) == ProjectionResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail
