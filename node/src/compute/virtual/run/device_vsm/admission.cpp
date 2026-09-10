#include "../../../../accel/kernel/prepared/run.hpp"
#include "internal.hpp"

#include "admission/pipeline.hpp"
#include "admission/shape.hpp"
#include "admission/validate.hpp"

#include "../../../type.hpp"

#include <kernel/program/compute/window/model.hpp>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

[[nodiscard]] constexpr ::rund::AccelCheck
fail(const char *const reason) noexcept {
  return {false, reason};
}

} // namespace

::rund::AccelCheck
admit_virtual_device_vsm(const VirtualPipelineState &state,
                         const VirtualRunProjection &run,
                         const VirtualDeviceVsmRouteProof &proof) noexcept {
  const std::uint64_t page_count = run.graph_execution()
                                       ? run.active.graph.page_count()
                                       : run.active.stream.page_count();
  if (!proof_matches(state, run, proof)) {
    return fail("compute_artifact_mismatch");
  }
  if (proof.page_count() != page_count ||
      proof.frame_capacity() != run.frame_capacity) {
    return fail("compute_shape_mismatch");
  }

  const bool window_ring =
      proof.kind() == VirtualDeviceVsmRouteKind::WindowRing;
  if (window_ring &&
      (state.pipeline == nullptr || state.alternate_pipeline == nullptr)) {
    return fail("compute_pipeline_invalid");
  }
  if (window_ring) {
    const ::rund::AccelCheck capability = node::accel::detail::
        QueryPreparedKernelPipelineDeviceVsmWindowCapability(
            state.pipeline->prepared, state.alternate_pipeline->prepared,
            type_scalar(run.input_type), type_domain(run.input_type),
            rund::kernel::WindowElement::U32);
    if (!capability.ok) {
      return capability;
    }
  }

  const std::size_t element_bytes = type_bytes(run.input_type);
  const admission_detail::Shape shape =
      admission_detail::classify(state, run, element_bytes);
  const std::shared_ptr<PipelineState> primary =
      state.device_vsm_semantic_pipeline != nullptr
          ? state.device_vsm_semantic_pipeline
          : state.pipeline;
  const std::shared_ptr<PipelineState> selected_primary =
      shape.graph_map_reduce ? primary : state.pipeline;
  const std::shared_ptr<PipelineState> second =
      shape.graph_map_reduce || shape.graph_pointwise
          ? graph_terminal_pipeline(state, 0u)
          : state.alternate_pipeline;
  const bool authority = admission_detail::pipeline_authority(
      selected_primary, second, run, shape.multi_pointwise, shape.multi_scan);

  if (!(shape.pointwise || shape.window || window_ring ||
        shape.graph_pointwise || shape.graph_map_reduce || shape.scan ||
        shape.reduce)) {
    return fail("compute_primitive_route_invalid");
  }
  if (!(shape.graph_pointwise || shape.graph_map_reduce || shape.scan ||
        shape.reduce ||
        (!run.reduction() && !run.graph_reduction() && !run.scan()))) {
    return fail("compute_primitive_route_invalid");
  }
  return admission_detail::check_bindings(
      run, page_count, element_bytes, shape.graph_pointwise,
      shape.graph_map_reduce, shape.scan, shape.reduce, selected_primary,
      second, authority);
}

} // namespace rund::compute::detail::device_vsm_product_detail
