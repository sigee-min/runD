#include "identity_internal.hpp"

#include "../../step/map/stride.hpp"

#include <cstddef>

namespace rund::node::accel::detail::backend_template_plan {

bool same_program_template(const KernelExecution &execution,
                           const PreparedKernelProgramRoute &left,
                           const PreparedKernelProgramRoute &right,
                           const std::uint64_t storage_alignment) noexcept {
  const rund::AccelKernel *const a = left.kernel;
  const rund::AccelKernel *const b = right.kernel;
  return storage_alignment != 0u && a != nullptr && b != nullptr &&
         a->owner != nullptr && a->owner == b->owner &&
         a->kernel_id == b->kernel_id && a->context_id == b->context_id &&
         a->graph_id_hi == b->graph_id_hi && a->graph_id_lo == b->graph_id_lo &&
         a->node_count == b->node_count && a->api == b->api &&
         a->scalar == b->scalar && a->domain == b->domain &&
         left.tile_count == right.tile_count &&
         same_layout(left.views, right.views) &&
         same_layout(left.scratch, right.scratch) &&
         same_program_map_specialization(execution, left, right,
                                         storage_alignment);
}

bool same_ref_layout(const rund::kernel::ResidentBindingRange &left,
                     const rund::kernel::ResidentBindingRange &right,
                     const std::uint64_t alignment,
                     const rund::kernel::ComputeApi api) noexcept {
  if (alignment == 0u || left.count != right.count) {
    return false;
  }
  for (std::uint64_t index = 0u; index < left.count; ++index) {
    const rund::kernel::ResidentBufferRef *const a = left.ref(index);
    const rund::kernel::ResidentBufferRef *const b = right.ref(index);
    if (a == nullptr || b == nullptr || a->element_bytes != b->element_bytes ||
        a->stride_bytes != b->stride_bytes || a->count != b->count ||
        a->usage != b->usage ||
        a->offset_bytes % alignment != b->offset_bytes % alignment ||
        (api == rund::kernel::ComputeApi::Metal &&
         MetalMapBindingWordClass(a->offset_bytes, a->stride_bytes) !=
             MetalMapBindingWordClass(b->offset_bytes, b->stride_bytes))) {
      return false;
    }
  }
  return true;
}

bool same_map_layout(const BoundStep &left, const BoundStep &right,
                     const std::uint64_t alignment,
                     const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::BindingSet a = MapBindingFor(left);
  const rund::kernel::BindingSet b = MapBindingFor(right);
  return a.ok && b.ok && a.input_buffer_count == b.input_buffer_count &&
         a.output_buffer_count == b.output_buffer_count &&
         same_ref_layout(a.resident_inputs, b.resident_inputs, alignment,
                         api) &&
         same_ref_layout(a.resident_outputs, b.resident_outputs, alignment,
                         api);
}

bool same_template(const BackendRun &left, const BackendRun &right,
                   const std::uint64_t alignment) noexcept {
  if (left.pick == nullptr || right.pick == nullptr ||
      left.pick != right.pick || left.steps == nullptr ||
      right.steps == nullptr || left.step_count == 0u ||
      left.step_count != right.step_count ||
      !same_layout(left.views, right.views) ||
      !same_layout(left.scratch, right.scratch)) {
    return false;
  }
  for (std::size_t index = 0u; index < left.step_count; ++index) {
    const BoundStep &a = left.steps[index];
    const BoundStep &b = right.steps[index];
    if (a.step == nullptr || b.step == nullptr || a.step != b.step ||
        a.planned == nullptr || b.planned == nullptr ||
        a.planned->domain != b.planned->domain ||
        a.planned->artifact != b.planned->artifact ||
        !same_plan(a.planned->plan, b.planned->plan) ||
        !same_windows(a.planned->windows, b.planned->windows) ||
        a.control.active() != b.control.active()) {
      return false;
    }
    if (a.step->kind() == rund::kernel::NodeKind::Map &&
        !same_map_layout(a, b, alignment, a.planned->plan.api)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail::backend_template_plan
