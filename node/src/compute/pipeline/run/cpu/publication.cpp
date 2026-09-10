#include "internal.hpp"

#include "../../state.hpp"

#include "../../../cpu/view.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>

namespace rund::compute::detail {

Status resolve_cpu_pipeline_publication_view(
    const CpuPipelinePublicationContext &context,
    const PipelinePublicationViewPlan &planned, CpuView &view) noexcept {
  const PipelinePublicationViewIdentity &identity = planned.identity;
  const bool alternate = context.alternate;
  const PipelineResource *const resource = selected_pipeline_resource(
      context.resources, identity.resource_ordinal, alternate);
  if (resource == nullptr || resource->buffer == nullptr ||
      resource->type != planned.type || resource->format != planned.format ||
      resource->buffer->type != planned.type ||
      resource->bytes != identity.backing_bytes ||
      identity.element_bytes == 0u ||
      identity.offset_bytes % identity.element_bytes != 0u ||
      identity.stride_bytes % identity.element_bytes != 0u ||
      identity.offset_bytes / identity.element_bytes >
          std::numeric_limits<std::size_t>::max() ||
      identity.count > std::numeric_limits<std::size_t>::max() ||
      identity.stride_bytes / identity.element_bytes >
          std::numeric_limits<std::size_t>::max()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::optional<CpuView> resolved = cpu_view(
      resource->buffer.get(),
      static_cast<std::size_t>(identity.offset_bytes / identity.element_bytes),
      static_cast<std::size_t>(identity.count),
      static_cast<std::size_t>(identity.stride_bytes / identity.element_bytes),
      static_cast<std::size_t>(identity.element_bytes));
  if (!resolved || (identity.count != 0u && resolved->data == nullptr) ||
      resolved->footprint.base != identity.offset_bytes ||
      resolved->footprint.count != identity.count ||
      resolved->footprint.stride != identity.stride_bytes ||
      resolved->footprint.width != identity.element_bytes) {
    return Status::fail(Reason::PipelineInvalid);
  }
  view = *resolved;
  return Status::success();
}

Status seal_cpu_resident(const CpuPipelinePublicationContext &context,
                         const std::size_t state_index,
                         const PipelineWindow &window,
                         const PipelineWindowProgress &progress) noexcept {
  if (window.first_step >= context.step_count ||
      progress.current > PipelineWindow::second ||
      window.recurrent_output_count == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t final = window.control.final;
  if (final < PipelineWindow::first || final > PipelineWindow::second) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint32_t sealed_count = 0u;
  for (const PipelinePublicationPlan &publication : context.publications) {
    const auto *terminal =
        std::get_if<PipelineTerminalPublicationPlan>(&publication);
    if (terminal == nullptr || terminal->state != state_index) {
      continue;
    }
    if (progress.current == final) {
      ++sealed_count;
      continue;
    }
    CpuView source{};
    CpuView target{};
    const Status source_ready = resolve_cpu_pipeline_publication_view(
        context, terminal->sources[progress.current], source);
    const Status target_ready = resolve_cpu_pipeline_publication_view(
        context, terminal->sources[final], target);
    if (!source_ready || !target_ready ||
        source.footprint.bytes != target.footprint.bytes ||
        source.footprint.stride != target.footprint.stride ||
        source.footprint.width != target.footprint.width ||
        source.data == nullptr || target.data == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (source.footprint.dense()) {
      std::memmove(target.data, source.data, source.footprint.bytes);
      ++sealed_count;
      continue;
    }
    const std::byte *read = source.data;
    std::byte *write = target.data;
    for (std::size_t remaining = source.footprint.count; remaining > 1u;
         --remaining) {
      std::memmove(write, read, source.footprint.width);
      read += source.footprint.stride;
      write += target.footprint.stride;
    }
    if (source.footprint.count != 0u) {
      std::memmove(write, read, source.footprint.width);
    }
    ++sealed_count;
  }
  return sealed_count == window.recurrent_output_count
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail
