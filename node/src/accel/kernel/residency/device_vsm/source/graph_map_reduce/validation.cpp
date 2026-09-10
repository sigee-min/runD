#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

bool validate(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &map_semantic,
    const rund::kernel::ReducePlan &reduce_semantic) noexcept {
  if (!device_vsm_typed_map::validate_total_u64(source, input, map_semantic) ||
      !reduce_semantic.ok ||
      reduce_semantic.element != rund::kernel::ReduceElement::U64 ||
      (reduce_semantic.op != rund::kernel::ReduceOp::Sum &&
       reduce_semantic.op != rund::kernel::ReduceOp::CountNonzero &&
       reduce_semantic.op != rund::kernel::ReduceOp::Min &&
       reduce_semantic.op != rund::kernel::ReduceOp::Max) ||
      reduce_semantic.count_source !=
          rund::kernel::ComputeCountSource::BufferU64) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
