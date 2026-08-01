#include "../local.hpp"
#include "../state.hpp"

#include "../../cpu/view.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace rund::compute::detail {

Status publish_cpu_pipeline(PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (const PipelinePublish &publication : state.publications) {
    if (publication.kind == PipelinePublishKind::Window) {
      continue;
    }
    if (publication.source == publication.target || publication.window == 0u ||
        publication.window > state.windows.size() ||
        publication.kind != PipelinePublishKind::Terminal ||
        publication.resident_count != nullptr || publication.maximum != 0u ||
        publication.tile != 0u) {
      return Status::fail(Reason::PipelineInvalid);
    }
    CpuView source{};
    const Status selected =
        cpu_resident_view(state, state.windows[publication.window - 1u],
                          publication.output, source);
    const std::optional<CpuView> target = cpu_view(
        publication.target.get(), publication.target_offset, publication.count,
        publication.target_stride, publication.element_bytes);
    if (!selected || !target ||
        source.footprint.bytes != target->footprint.bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bytes = source.footprint.bytes;
    if (bytes == 0u) {
      continue;
    }
    if (source.data == nullptr || target->data == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (target->footprint.dense()) {
      std::memcpy(target->data, source.data, bytes);
      continue;
    }

    const std::byte *from = source.data;
    std::byte *to = target->data;
    for (std::size_t remaining = publication.count; remaining > 1u;
         --remaining) {
      std::memcpy(to, from, source.footprint.width);
      from += source.footprint.stride;
      to += target->footprint.stride;
    }
    std::memcpy(to, from, source.footprint.width);
  }
  return Status::success();
}

Status publish_cpu_pipeline_window(PipelineState &state,
                                   const std::uint16_t window,
                                   const std::size_t outer,
                                   bool &wrote) noexcept {
  wrote = false;
  if (state.device == nullptr || state.device->backend != Backend::Cpu ||
      window == 0u || window > state.windows.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineWindow &descriptor = state.windows[window - 1u];
  if (!descriptor.nested || descriptor.tile == 0u || descriptor.maximum == 0u ||
      descriptor.tile > descriptor.maximum || outer >= descriptor.seed_count) {
    return Status::fail(Reason::PipelineInvalid);
  }

  for (const PipelinePublish &publication : state.publications) {
    if (publication.kind != PipelinePublishKind::Window ||
        publication.window != window) {
      continue;
    }
    if (publication.source == nullptr || publication.target == nullptr ||
        publication.resident_count == nullptr ||
        publication.source == publication.target ||
        publication.maximum != descriptor.maximum ||
        publication.tile != descriptor.tile ||
        publication.count != publication.tile ||
        publication.target_stride != 1u || publication.element_bytes == 0u) {
      return Status::fail(Reason::PipelineInvalid);
    }

    const std::optional<CpuView> count = cpu_view(
        publication.resident_count.get(), publication.resident_count_offset, 1u,
        1u, sizeof(std::uint32_t));
    if (!count || count->data == nullptr || count->footprint.count != 1u) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint32_t resident_count{};
    std::memcpy(&resident_count, count->data, sizeof(resident_count));
    if (resident_count > publication.maximum) {
      return Status::fail(Reason::BoundedCountInvalid);
    }

    const std::uint64_t base =
        static_cast<std::uint64_t>(outer) * publication.tile;
    if (base >= resident_count || base >= publication.maximum) {
      continue;
    }
    const std::size_t active = static_cast<std::size_t>(
        std::min({static_cast<std::uint64_t>(publication.tile),
                  static_cast<std::uint64_t>(resident_count) - base,
                  static_cast<std::uint64_t>(publication.maximum) - base}));
    if (active == 0u || base > std::numeric_limits<std::size_t>::max() -
                                   publication.target_offset) {
      return Status::fail(Reason::PipelineInvalid);
    }

    const std::optional<CpuView> source =
        cpu_view(publication.source.get(), publication.source_offset,
                 publication.count, 1u, publication.element_bytes);
    const std::optional<CpuView> target =
        cpu_view(publication.target.get(),
                 publication.target_offset + static_cast<std::size_t>(base),
                 active, 1u, publication.element_bytes);
    if (!source || !target || source->data == nullptr ||
        target->data == nullptr ||
        source->footprint.count != publication.tile ||
        source->footprint.width != publication.element_bytes ||
        !source->footprint.dense() || !target->footprint.dense()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::memcpy(target->data, source->data, active * publication.element_bytes);
    wrote = true;
  }
  return Status::success();
}

} // namespace rund::compute::detail
