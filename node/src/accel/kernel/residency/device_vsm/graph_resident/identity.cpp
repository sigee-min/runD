#include "../graph_resident.hpp"

namespace rund::node::accel::detail {

bool device_vsm_graph_resident_same_object(
    const std::shared_ptr<void> &left,
    const std::shared_ptr<void> &right) noexcept {
  return left != nullptr && right != nullptr && left.get() == right.get() &&
         !left.owner_before(right) && !right.owner_before(left);
}

} // namespace rund::node::accel::detail
