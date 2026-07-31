#include "local.hpp"

namespace rund::node::accel::detail {

rund::kernel::FusionPolicy FusionPolicyFor(
    const std::vector<rund::kernel::FusionNodePolicy>& nodes) noexcept {
  return rund::kernel::FusionPolicy{
      .nodes = nodes.empty() ? nullptr : nodes.data(),
      .node_count = static_cast<rund::kernel::u64>(nodes.size()),
  };
}

}  // namespace rund::node::accel::detail
