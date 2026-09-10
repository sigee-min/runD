#include "internal.hpp"

#include "../output.hpp"
#include "../../size.hpp"
#include "../../type.hpp"

#include <rund/compute/resource/plan.hpp>

#include <limits>

namespace rund::compute::detail {

PipelineBinding bind(const ResourceView &view, const bool hidden) noexcept {
  return PipelineBinding{.buffer = view.buffer,
                         .type = view.type,
                         .format = view.format,
                         .offset = view.offset,
                         .count = view.count,
                         .stride = view.stride,
                         .element_bytes = view.element_bytes,
                         .alignment = view.alignment,
                         .backing_bytes =
                             view.buffer == nullptr ? 0u : view.buffer->bytes,
                         .access = view.access,
                         .hidden = hidden};
}

PipelineBinding bind(const std::uint32_t owner,
                     const PipelineInternal &resource,
                     const ResourceAccess access, const std::size_t offset,
                     const std::size_t count, const bool hidden) noexcept {
  const std::size_t elements = count == 0u ? resource.count : count;
  const std::size_t width = type_bytes(resource.type);
  const std::size_t backing =
      width != 0u &&
              resource.count <= std::numeric_limits<std::size_t>::max() / width
          ? resource.count * width
          : 0u;
  return PipelineBinding{.type = resource.type,
                         .format = resource.format,
                         .offset = offset,
                         .count = elements,
                         .stride = 1u,
                         .element_bytes = width,
                         .alignment = width,
                         .backing_bytes = backing,
                         .access = access,
                         .owner = owner,
                         .hidden = hidden};
}

Result<bool> intersects(const PipelineBinding &left,
                        const ResourceView &right) noexcept {
  if (left.buffer == nullptr || right.buffer == nullptr ||
      left.buffer != right.buffer || left.element_bytes == 0u ||
      right.element_bytes == 0u) {
    return Result<bool>::success(false);
  }
  const auto scaled = [](const std::size_t value,
                         const std::size_t width) noexcept {
    return value > std::numeric_limits<std::uint64_t>::max() / width
               ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(value) * width;
  };
  const resource::Resource owner{
      .id = 1u,
      .bytes = left.buffer->bytes,
      .alias_group = 1u,
  };
  const resource::Access a{
      .resource = 1u,
      .offset_bytes = scaled(left.offset, left.element_bytes),
      .element_bytes = left.element_bytes,
      .element_count = left.count,
      .stride_bytes = scaled(left.stride, left.element_bytes),
  };
  const resource::Access b{
      .resource = 1u,
      .offset_bytes = scaled(right.offset, right.element_bytes),
      .element_bytes = right.element_bytes,
      .element_count = right.count,
      .stride_bytes = scaled(right.stride, right.element_bytes),
  };
  return resource::intersects(owner, a, owner, b);
}

bool same_view(const ResourceView &left, const ResourceView &right) noexcept {
  return left.buffer == right.buffer && left.type == right.type &&
         left.format == right.format && left.offset == right.offset &&
         left.count == right.count && left.stride == right.stride &&
         left.element_bytes == right.element_bytes &&
         left.alignment == right.alignment && left.access == right.access;
}

Status route(const PipelineBuildState &build, const ResourceView &view,
             PipelineBinding &result) noexcept {
  result = bind(view);
  for (const PipelineBuildPublication &publication : build.publications) {
    const PipelineBuildPublicationEdge &edge =
        pipeline_publication_edge(publication);
    const PipelineBinding &target = edge.target;
    if (target.buffer != view.buffer) {
      continue;
    }
    auto overlap = intersects(target, view);
    if (!overlap) {
      return Status::fail(overlap.reason());
    }
    if (!*overlap) {
      if (view.access == ResourceAccess::Write) {
        // Publication output identity is currently Buffer-owned.  A second
        // disjoint public write would otherwise be accepted here and rejected
        // later as a duplicate output owner.  Reject it at the authored edge
        // instead of allowing terminal publication to clobber or obscure it.
        return Status::fail(Reason::BindingAliasUnsupported);
      }
      continue;
    }
    if (std::holds_alternative<PipelineBuildWindowPublication>(publication)) {
      if (view.access != ResourceAccess::Read) {
        // The append-only target has one writer for the complete Pipeline.
        return Status::fail(Reason::BindingAliasUnsupported);
      }
      // Declaration order has already sealed the complete nested window.
      // Unlike terminal recurrence, the O(Tile) private source cannot stand
      // in for the O(Max) result. Keep the caller target as the read binding;
      // resource analysis then owns the exact publication-to-read barrier.
      return Status::success();
    }
    const auto source_coordinate = resolve_publication_source(
        build, std::get<PipelineBuildTerminalPublication>(publication));
    if (!source_coordinate) {
      return Status::fail(source_coordinate.reason());
    }
    const auto projected = resolve_build_output(build, *source_coordinate);
    if (!projected) {
      return Status::fail(projected.reason());
    }
    const PipelineBinding &source = build.steps[source_coordinate->step.value]
                                        .outputs[projected->source.value];
    if (target.type != view.type || target.format != view.format ||
        target.element_bytes != view.element_bytes || target.stride == 0u ||
        view.offset < target.offset) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    const std::size_t delta = view.offset - target.offset;
    if (delta % target.stride != 0u || view.stride % target.stride != 0u) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    const std::size_t first = delta / target.stride;
    const std::size_t step = view.stride / target.stride;
    std::size_t distance = 0u;
    std::size_t last = first;
    const bool tail_overflow =
        view.count != 0u && step != 0u &&
        (!size::multiply(view.count - 1u, step, distance) ||
         !size::add(first, distance, last));
    if (source.stride == 0u || step == 0u || first >= target.count ||
        tail_overflow || (view.count != 0u && last >= target.count)) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    std::size_t source_delta = 0u;
    std::size_t source_offset = 0u;
    std::size_t source_stride = 0u;
    if (!size::multiply(first, source.stride, source_delta) ||
        !size::add(source.offset, source_delta, source_offset) ||
        !size::multiply(step, source.stride, source_stride)) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    result = source;
    result.offset = source_offset;
    result.count = view.count;
    result.stride = source_stride;
    result.alignment = view.alignment;
    result.access = view.access;
    result.hidden = true;
    return Status::success();
  }
  return Status::success();
}

void changed(PipelineBuildState &build) noexcept { build.memory.reset(); }

bool has_seed(const PipelineBuildState &build) noexcept {
  return build.seed != nullptr || build.storage_seed != nullptr ||
         build.device_seed != nullptr;
}

} // namespace rund::compute::detail
