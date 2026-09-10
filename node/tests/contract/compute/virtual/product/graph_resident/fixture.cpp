#include "internal.hpp"

#include <span>

namespace rund_node_test_virtual::product::graph_resident {

bool seed_backing(const std::shared_ptr<rund::compute::VirtualBacking> &backing,
                  const std::span<const std::byte> values) noexcept {
  return backing != nullptr && static_cast<bool>(backing->write(0u, values));
}
} // namespace rund_node_test_virtual::product::graph_resident
