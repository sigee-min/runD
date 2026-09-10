#include "local.hpp"

namespace rund_node_test_compute_reuse {

bool SameMemory(const rund::compute::MemoryStats &left,
                const rund::compute::MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         left.host.current == right.host.current &&
         left.frame.current == right.frame.current &&
         left.tile.current == right.tile.current &&
         left.resident.current == right.resident.current &&
         left.staging.current == right.staging.current &&
         left.device.current == right.device.current &&
         left.transfer.current == right.transfer.current;
}

} // namespace rund_node_test_compute_reuse
