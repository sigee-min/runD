#include "resource.hpp"

#include "contract.hpp"

#include "../../size.hpp"
#include "../../type.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace rund::compute::detail {

PipelineScheduleResources::PipelineScheduleResources(
    const PipelineBuildState &declared)
    : build(declared),
      internals_(build.internals.size(), PipelineResourceUnassigned) {
  shapes.reserve(std::min(build.binding_count, PipelineResourceCapacity));
  resources.reserve(std::min(build.binding_count, PipelineResourceCapacity));
  use_evidence.reserve(std::min(build.binding_count, PipelineResourceCapacity));
  accesses.reserve(build.binding_count);
  external_flags.reserve(
      std::min(build.binding_count, PipelineResourceCapacity));
  publication_accesses.reserve(build.publications.size() * 2u);
  external_.reserve(std::min(build.binding_count, PipelineResourceCapacity));
}

Result<std::uint32_t>
PipelineScheduleResources::admit(const PipelineBinding &binding,
                                 const Type slot_type,
                                 const FixedFormat slot_format) {
  const BufferState *external_buffer = nullptr;
  std::uint32_t *internal_ordinal = nullptr;
  PipelineResolvedResourcePlan resolved{
      .locator = PipelineExternalResourcePlan{},
      .type = slot_type,
      .format = slot_format,
  };
  if (binding.owner == PipelineBinding::external) {
    if (binding.buffer == nullptr) {
      return Result<std::uint32_t>::fail(Reason::BindingInvalid);
    }
    external_buffer = binding.buffer.get();
    const auto found = external_.find(external_buffer);
    if (found != external_.end()) {
      return Result<std::uint32_t>::success(found->second);
    }
    resolved.locator = PipelineExternalResourcePlan{.owner = binding.buffer};
    resolved.count = binding.buffer->count;
    resolved.bytes = binding.buffer->bytes;
    resolved.physical_bytes = binding.buffer->physical_bytes;
  } else {
    if (binding.owner >= build.internals.size()) {
      return Result<std::uint32_t>::fail(Reason::PipelineInvalid);
    }
    internal_ordinal = &internals_[binding.owner];
    if (*internal_ordinal != PipelineResourceUnassigned) {
      return Result<std::uint32_t>::success(*internal_ordinal);
    }
    const PipelineInternal &internal = build.internals[binding.owner];
    std::size_t internal_bytes = 0u;
    const std::size_t width = type_bytes(internal.type);
    if (width == 0u || !size::multiply(internal.count, width, internal_bytes)) {
      return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
    }
    resolved.locator = PipelineInternalResourcePlan{.fill = internal.fill};
    resolved.count = internal.count;
    resolved.bytes = internal_bytes;
    resolved.physical_bytes = internal_bytes;
  }
  if (resources.size() >= PipelineResourceCapacity) {
    return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
  }
  const auto ordinal = static_cast<std::uint32_t>(resources.size());
  resources.push_back(std::move(resolved));
  use_evidence.emplace_back();
  shapes.push_back(resource::Resource{
      .id = ordinal + 1u,
      .bytes = resources.back().bytes,
      .alias_group = ordinal + 1u,
  });
  external_flags.push_back(external_buffer != nullptr ? 1u : 0u);
  if (internal_ordinal != nullptr) {
    *internal_ordinal = ordinal;
  } else {
    external_.emplace(external_buffer, ordinal);
  }
  return Result<std::uint32_t>::success(ordinal);
}

Result<PipelineResolvedViewPlan> PipelineScheduleResources::resolve(
    const PipelineBinding &binding, const Type slot_type,
    const FixedFormat slot_format, const Location location) {
  auto ordinal = admit(binding, slot_type, slot_format);
  if (!ordinal) {
    return Result<PipelineResolvedViewPlan>::fail(ordinal.reason(), location);
  }
  if (*ordinal >= resources.size()) {
    return Result<PipelineResolvedViewPlan>::fail(Reason::PipelineInvalid,
                                                  location);
  }
  const PipelineResolvedResourcePlan &resource = resources[*ordinal];
  const std::size_t element_bytes = binding.element_bytes;
  if (element_bytes == 0u || binding.stride == 0u ||
      resource.bytes > std::numeric_limits<std::size_t>::max() ||
      resource.physical_bytes < resource.bytes) {
    return Result<PipelineResolvedViewPlan>::fail(Reason::ShapeMismatch,
                                                  location);
  }

  const auto owner_bytes = static_cast<std::size_t>(resource.bytes);
  std::size_t offset_bytes = 0u;
  std::size_t stride_bytes = 0u;
  std::size_t payload_bytes = 0u;
  std::size_t span_bytes = 0u;
  if (!size::multiply(binding.offset, element_bytes, offset_bytes) ||
      !size::multiply(binding.stride, element_bytes, stride_bytes) ||
      !size::multiply(binding.count, element_bytes, payload_bytes)) {
    return Result<PipelineResolvedViewPlan>::fail(Reason::ShapeMismatch,
                                                  location);
  }
  const bool span_overflow =
      binding.count != 0u &&
      (!size::multiply(binding.count - 1u, stride_bytes, span_bytes) ||
       !size::add(span_bytes, element_bytes, span_bytes));
  if (span_overflow || offset_bytes > owner_bytes ||
      (binding.count != 0u && (span_bytes > owner_bytes - offset_bytes))) {
    return Result<PipelineResolvedViewPlan>::fail(Reason::ShapeMismatch,
                                                  location);
  }

  return Result<PipelineResolvedViewPlan>::success(PipelineResolvedViewPlan{
      .resource = *ordinal,
      .declared_type = binding.type,
      .declared_format = binding.format,
      .declared_access = binding.access,
      .declared_backing_bytes = binding.backing_bytes,
      .offset = binding.offset,
      .count = binding.count,
      .stride = binding.stride,
      .element_bytes = element_bytes,
      .alignment = binding.alignment,
      .offset_bytes = offset_bytes,
      .stride_bytes = stride_bytes,
      .payload_bytes = payload_bytes,
      .span_bytes = binding.count == 0u ? 0u : span_bytes,
  });
}

Result<PipelinePublicationViewPlan> PipelineScheduleResources::publication_view(
    const PipelineBinding &binding, const Type slot_type,
    const std::size_t slot_count, const FixedFormat slot_format,
    const std::optional<ResourceAccess> expected_access,
    const std::uint32_t usage, const Location location) {
  if (usage != rund::kernel::kResidentUsageRead &&
      usage != rund::kernel::kResidentUsageWrite) {
    return Result<PipelinePublicationViewPlan>::fail(Reason::PipelineInvalid,
                                                     location);
  }
  auto view = resolve(binding, slot_type, slot_format, location);
  if (!view) {
    return Result<PipelinePublicationViewPlan>::fail(view.reason(),
                                                     view.location());
  }
  const PipelineResolvedResourcePlan &resource = resources[view->resource];
  bool owner_matches = false;
  if (binding.owner == PipelineBinding::external) {
    const auto *external =
        std::get_if<PipelineExternalResourcePlan>(&resource.locator);
    owner_matches = external != nullptr && external->owner != nullptr &&
                    external->owner == binding.buffer &&
                    external->owner->type == slot_type &&
                    external->owner->bytes == resource.bytes &&
                    external->owner->physical_bytes >= external->owner->bytes;
  } else if (binding.owner < build.internals.size()) {
    const PipelineInternal &internal = build.internals[binding.owner];
    owner_matches = std::holds_alternative<PipelineInternalResourcePlan>(
                        resource.locator) &&
                    internal.type == slot_type &&
                    internal.format == slot_format &&
                    internal.count == resource.count;
  }
  if ((expected_access.has_value() &&
       view->declared_access != *expected_access) ||
      !valid_type(slot_type) || view->declared_type != slot_type ||
      resource.type != slot_type || resource.format != slot_format ||
      !owner_matches || view->count != slot_count ||
      view->element_bytes != type_bytes(slot_type) || view->alignment == 0u ||
      (view->alignment & (view->alignment - 1u)) != 0u ||
      view->alignment > 64u || view->offset_bytes % view->alignment != 0u ||
      view->declared_backing_bytes != resources[view->resource].bytes ||
      !typed_format_matches(slot_type, view->declared_format, slot_format)) {
    return Result<PipelinePublicationViewPlan>::fail(
        expected_access.has_value() && view->declared_access != *expected_access
            ? Reason::BindingInvalid
        : !valid_type(slot_type) || view->declared_type != slot_type
            ? Reason::BindingTypeMismatch
        : !typed_format_matches(slot_type, view->declared_format, slot_format)
            ? (valid_format(slot_type, slot_format)
                   ? Reason::FixedFormatMismatch
                   : Reason::FixedFormatInvalid)
            : Reason::ShapeMismatch,
        location);
  }
  return publication_view(*view, usage, location);
}

Result<PipelinePublicationViewPlan> PipelineScheduleResources::publication_view(
    const PipelineResolvedViewPlan &view, const std::uint32_t usage,
    const Location location) const {
  if ((usage != rund::kernel::kResidentUsageRead &&
       usage != rund::kernel::kResidentUsageWrite) ||
      view.resource >= resources.size()) {
    return Result<PipelinePublicationViewPlan>::fail(Reason::PipelineInvalid,
                                                     location);
  }
  const PipelineResolvedResourcePlan &resource = resources[view.resource];
  return Result<PipelinePublicationViewPlan>::success(
      PipelinePublicationViewPlan{
          .identity =
              PipelinePublicationViewIdentity{
                  .backing_bytes = resource.bytes,
                  .offset_bytes = view.offset_bytes,
                  .count = view.count,
                  .stride_bytes = view.stride_bytes,
                  .element_bytes = view.element_bytes,
                  .resource_ordinal = view.resource,
                  .usage = usage,
              },
          .type = view.declared_type,
          .format = view.declared_format,
          .offset = view.offset,
          .stride = view.stride,
          .alignment = view.alignment,
      });
}

Status PipelineScheduleResources::complete_internal_resources() {
  for (std::size_t index = 0u; index < build.internals.size(); ++index) {
    if (internals_[index] != PipelineResourceUnassigned) {
      continue;
    }
    const PipelineInternal &internal = build.internals[index];
    const std::size_t element_bytes = type_bytes(internal.type);
    std::size_t backing_bytes = 0u;
    if (element_bytes == 0u ||
        !size::multiply(internal.count, element_bytes, backing_bytes) ||
        index >= PipelineBinding::external) {
      return Status::fail(Reason::PipelineCapacity);
    }
    const PipelineBinding binding{
        .type = internal.type,
        .format = internal.format,
        .count = internal.count,
        .stride = 1u,
        .element_bytes = element_bytes,
        .alignment = element_bytes,
        .backing_bytes = backing_bytes,
        .access = ResourceAccess::Read,
        .owner = static_cast<std::uint32_t>(index),
    };
    auto ordinal = admit(binding, internal.type, internal.format);
    if (!ordinal || *ordinal != internals_[index]) {
      return Status::fail(ordinal ? Reason::PipelineInvalid : ordinal.reason());
    }
  }
  return Status::success();
}

bool PipelineScheduleResources::append(
    std::vector<resource::Access> &destination,
    const PipelineResolvedViewPlan &view, const std::uint32_t node,
    const resource::AccessMode mode) {
  if (view.count == 0u) {
    return true;
  }
  if (view.resource == PipelineResourceUnassigned || view.element_bytes == 0u ||
      view.stride_bytes == 0u) {
    return false;
  }
  destination.push_back(resource::Access{
      .node = node,
      .resource = view.resource + 1u,
      .mode = mode,
      .offset_bytes = view.offset_bytes,
      .element_bytes = view.element_bytes,
      .element_count = view.count,
      .stride_bytes = view.stride_bytes,
  });
  return true;
}

bool PipelineScheduleResources::append(
    std::vector<resource::Access> &destination,
    const PipelinePublicationViewIdentity &view, const std::uint32_t node) {
  resource::AccessMode mode{};
  if (view.usage == rund::kernel::kResidentUsageRead) {
    mode = resource::AccessMode::Read;
  } else if (view.usage == rund::kernel::kResidentUsageWrite) {
    mode = resource::AccessMode::Write;
  } else {
    return false;
  }
  if (view.count == 0u) {
    return true;
  }
  if (view.resource_ordinal == PipelineResourceUnassigned ||
      view.element_bytes == 0u || view.stride_bytes == 0u) {
    return false;
  }
  destination.push_back(resource::Access{
      .node = node,
      .resource = view.resource_ordinal + 1u,
      .mode = mode,
      .offset_bytes = view.offset_bytes,
      .element_bytes = view.element_bytes,
      .element_count = view.count,
      .stride_bytes = view.stride_bytes,
  });
  return true;
}

} // namespace rund::compute::detail
