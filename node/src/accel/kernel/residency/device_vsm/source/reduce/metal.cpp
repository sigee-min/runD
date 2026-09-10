#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_reduce_source {

std::string metal_source(const rund::kernel::ArtifactKey &key,
                         const rund::kernel::ReduceElement element,
                         const rund::kernel::ReduceOp operation) {
  if (operation == rund::kernel::ReduceOp::Sum ||
      operation == rund::kernel::ReduceOp::CountNonzero) {
    return metal_additive_source(key, element, operation);
  }
  if (operation == rund::kernel::ReduceOp::Min ||
      operation == rund::kernel::ReduceOp::Max) {
    return metal_extreme_source(key, element, operation);
  }
  return {};
}

} // namespace rund::node::accel::detail::device_vsm_reduce_source
