#include "../../state/assembly.hpp"
#include "model.hpp"

#include "../../claim.hpp"
#include "../../local.hpp"
#include "../../../type.hpp"

#include <rund/compute/resource/plan.hpp>

namespace rund::compute::detail {

AdmissionDraft::AdmissionDraft(const PipelineBuildState &build_ref,
                               const PipelineMemoryPlan &plan_ref,
                               const PipelineBuildSnapshot &frozen_ref) noexcept
    : build(build_ref), plan(plan_ref), frozen(frozen_ref),
      resource_count(plan_ref.resources.size()),
      binding_capacity(frozen_ref.nested_windows.empty()
                           ? PipelineBindingCapacity
                           : PipelineRouteBindingCapacity) {}

namespace admit {

[[nodiscard]] bool publication_matches_resolved(
    const PipelinePublicationViewPlan &publication,
    const PipelineResolvedViewPlan &resolved) noexcept {
  return publication.identity.resource_ordinal == resolved.resource &&
         publication.identity.offset_bytes == resolved.offset_bytes &&
         publication.identity.count == resolved.count &&
         publication.identity.stride_bytes == resolved.stride_bytes &&
         publication.identity.element_bytes == resolved.element_bytes &&
         publication.type == resolved.declared_type &&
         publication.format == resolved.declared_format &&
         publication.offset == resolved.offset &&
         publication.stride == resolved.stride &&
         publication.alignment == resolved.alignment;
}

[[nodiscard]] Result<std::uint32_t>
admit_resolved_view(const PipelineMemoryPlan &plan, const PipelineState &state,
                    const PipelineResolvedViewPlan &view, const Type slot_type,
                    const std::size_t slot_count, const FixedFormat slot_format,
                    const ResourceAccess expected_access) noexcept {
  if (view.resource >= plan.resources.size() ||
      view.resource >= state.resources.size()) {
    return Result<std::uint32_t>::fail(Reason::PipelineInvalid);
  }
  const PipelineResolvedResourcePlan &planned = plan.resources[view.resource];
  const PipelineResource &resource = state.resources[view.resource];
  const std::shared_ptr<BufferState> &owner = resource.buffer;
  const bool external =
      std::holds_alternative<PipelineExternalResourcePlan>(planned.locator);
  if (owner == nullptr) {
    return Result<std::uint32_t>::fail(Reason::BindingInvalid);
  }
  const Status device = validate_pipeline_resource_device(state, resource);
  if (!device) {
    return Result<std::uint32_t>::fail(device.reason());
  }
  if (!valid_type(slot_type) || view.declared_type != slot_type ||
      planned.type != slot_type || resource.type != slot_type ||
      owner->type != slot_type) {
    return Result<std::uint32_t>::fail(Reason::BindingTypeMismatch);
  }
  if (!typed_format_matches(slot_type, view.declared_format, slot_format) ||
      planned.format != slot_format || resource.format != slot_format) {
    return Result<std::uint32_t>::fail(valid_format(slot_type, slot_format)
                                           ? Reason::FixedFormatMismatch
                                           : Reason::FixedFormatInvalid);
  }
  const std::size_t element_bytes = type_bytes(slot_type);
  if (view.declared_access != expected_access || element_bytes == 0u ||
      view.element_bytes != element_bytes || view.count != slot_count ||
      view.stride == 0u || view.alignment == 0u ||
      (view.alignment & (view.alignment - 1u)) != 0u || view.alignment > 64u ||
      view.offset_bytes % view.alignment != 0u ||
      view.declared_backing_bytes != planned.bytes ||
      planned.count != owner->count || planned.bytes != owner->bytes ||
      (external && planned.physical_bytes != owner->physical_bytes) ||
      owner->physical_bytes < owner->bytes ||
      view.offset_bytes > planned.bytes ||
      (view.count != 0u &&
       view.span_bytes > planned.bytes - view.offset_bytes)) {
    return Result<std::uint32_t>::fail(view.declared_access != expected_access
                                           ? Reason::BindingInvalid
                                           : Reason::ShapeMismatch);
  }
  return Result<std::uint32_t>::success(view.resource);
}

[[nodiscard]] PipelineResolvedViewPlan const *
physical_output_view(const PipelineStepResourcePlan &step,
                     const std::size_t physical) noexcept {
  if (physical >= step.physical_sources.size()) {
    return nullptr;
  }
  const std::uint32_t logical = step.physical_sources[physical];
  return logical < step.outputs.size() ? &step.outputs[logical].view : nullptr;
}

[[nodiscard]] Result<bool> resolved_views_intersect(
    const PipelineMemoryPlan &plan, const PipelineResolvedViewPlan &left,
    const resource::AccessMode left_mode, const PipelineResolvedViewPlan &right,
    const resource::AccessMode right_mode) noexcept {
  if (left.resource >= plan.resources.size() ||
      right.resource >= plan.resources.size()) {
    return Result<bool>::fail(Reason::PipelineInvalid);
  }
  const auto shape = [&](const PipelineResolvedViewPlan &view) {
    return resource::Resource{.id = view.resource + 1u,
                              .bytes = plan.resources[view.resource].bytes,
                              .alias_group = view.resource + 1u};
  };
  const auto access = [](const PipelineResolvedViewPlan &view,
                         const resource::AccessMode mode) {
    return resource::Access{.resource = view.resource + 1u,
                            .mode = mode,
                            .offset_bytes = view.offset_bytes,
                            .element_bytes = view.element_bytes,
                            .element_count = view.count,
                            .stride_bytes = view.stride_bytes};
  };
  return resource::intersects(shape(left), access(left, left_mode),
                              shape(right), access(right, right_mode));
}

[[nodiscard]] Status
validate_step_aliases(const PipelineMemoryPlan &plan,
                      const PipelineStepResourcePlan &step) noexcept {
  for (const PipelineResolvedViewPlan &input : step.inputs) {
    for (std::size_t physical = 0u; physical < step.physical_sources.size();
         ++physical) {
      const PipelineResolvedViewPlan *const output =
          physical_output_view(step, physical);
      if (output == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (input.resource != output->resource) {
        continue;
      }
      auto overlap =
          resolved_views_intersect(plan, input, resource::AccessMode::Read,
                                   *output, resource::AccessMode::Write);
      if (!overlap || *overlap) {
        return Status::fail(overlap ? Reason::BindingAliasUnsupported
                                    : overlap.reason());
      }
    }
  }
  for (std::size_t left = 0u; left < step.physical_sources.size(); ++left) {
    const PipelineResolvedViewPlan *const left_view =
        physical_output_view(step, left);
    if (left_view == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (std::size_t right = left + 1u; right < step.physical_sources.size();
         ++right) {
      const PipelineResolvedViewPlan *const right_view =
          physical_output_view(step, right);
      if (right_view == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (left_view->resource != right_view->resource) {
        continue;
      }
      auto overlap = resolved_views_intersect(
          plan, *left_view, resource::AccessMode::Write, *right_view,
          resource::AccessMode::Write);
      if (!overlap || *overlap) {
        return Status::fail(overlap ? Reason::BindingDuplicate
                                    : overlap.reason());
      }
    }
  }
  return Status::success();
}


} // namespace admit

} // namespace rund::compute::detail
