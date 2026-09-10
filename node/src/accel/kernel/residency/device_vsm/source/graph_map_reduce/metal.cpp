#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

std::string
metal_source(const rund::kernel::ArtifactKey &key,
             const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
             const rund::kernel::ReduceOp operation,
             const DeviceVsmGraphWavefrontProof &wavefront) {
  if (operation == rund::kernel::ReduceOp::Sum ||
      operation == rund::kernel::ReduceOp::CountNonzero) {
    return metal_additive_source(key, parsed, operation, wavefront);
  }
  return metal_extreme_source(key, parsed, operation, wavefront);
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
