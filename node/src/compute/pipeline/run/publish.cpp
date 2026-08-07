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
  for (const PipelinePublicationPlan &publication : state.publications) {
    const auto *terminal =
        std::get_if<PipelineTerminalPublicationPlan>(&publication);
    if (terminal == nullptr) {
      continue;
    }
    if (terminal->state >= state.windows.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineWindowControl &control =
        state.windows[terminal->state].control;
    if (control.final < PipelineWindow::first ||
        control.final > PipelineWindow::second) {
      return Status::fail(Reason::PipelineInvalid);
    }
    CpuView source{};
    CpuView target{};
    const Status selected = resolve_cpu_pipeline_publication_view(
        state, terminal->sources[control.final], source);
    const Status targeted = resolve_cpu_pipeline_publication_view(
        state, terminal->target.view, target);
    if (!selected || !targeted ||
        source.footprint.count != target.footprint.count ||
        source.footprint.width != target.footprint.width) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bytes = source.footprint.count * source.footprint.width;
    if (bytes == 0u) {
      continue;
    }
    if (source.data == nullptr || target.data == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (source.footprint.dense() && target.footprint.dense()) {
      std::memcpy(target.data, source.data, bytes);
      continue;
    }

    const std::byte *from = source.data;
    std::byte *to = target.data;
    for (std::size_t remaining = source.footprint.count; remaining > 1u;
         --remaining) {
      std::memcpy(to, from, source.footprint.width);
      from += source.footprint.stride;
      to += target.footprint.stride;
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
  const PipelineWindowControl &control = descriptor.control;
  if (!descriptor.nested() || control.tile == 0u || control.maximum == 0u ||
      control.tile > control.maximum ||
      outer >= descriptor.nested_shape.outer_bound()) {
    return Status::fail(Reason::PipelineInvalid);
  }

  for (const PipelinePublicationPlan &publication : state.publications) {
    const auto *planned =
        std::get_if<PipelineWindowPublicationPlan>(&publication);
    if (planned == nullptr || planned->state + 1u != window) {
      continue;
    }
    if (planned->state >= state.windows.size() ||
        planned->source.identity.count != control.tile ||
        planned->source.identity.element_bytes == 0u ||
        planned->target.view.identity.count != control.maximum) {
      return Status::fail(Reason::PipelineInvalid);
    }

    CpuView count{};
    const Status count_ready =
        resolve_cpu_pipeline_publication_view(state, control.count, count);
    if (!count_ready || count.data == nullptr || count.footprint.count != 1u ||
        count.footprint.width != sizeof(std::uint32_t)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint32_t resident_count{};
    std::memcpy(&resident_count, count.data, sizeof(resident_count));
    if (resident_count > control.maximum) {
      return Status::fail(Reason::BoundedCountInvalid);
    }

    const std::uint64_t base = static_cast<std::uint64_t>(outer) * control.tile;
    if (base >= resident_count || base >= control.maximum) {
      continue;
    }
    const std::size_t active = static_cast<std::size_t>(
        std::min({static_cast<std::uint64_t>(control.tile),
                  static_cast<std::uint64_t>(resident_count) - base,
                  static_cast<std::uint64_t>(control.maximum) - base}));
    if (active == 0u || base > std::numeric_limits<std::size_t>::max()) {
      return Status::fail(Reason::PipelineInvalid);
    }

    CpuView source{};
    CpuView target{};
    const Status source_ready =
        resolve_cpu_pipeline_publication_view(state, planned->source, source);
    const Status target_ready = resolve_cpu_pipeline_publication_view(
        state, planned->target.view, target);
    if (!source_ready || !target_ready || source.data == nullptr ||
        target.data == nullptr || source.footprint.count != control.tile ||
        source.footprint.width != planned->source.identity.element_bytes ||
        !source.footprint.dense() || !target.footprint.dense() ||
        base > target.footprint.count ||
        active > target.footprint.count - static_cast<std::size_t>(base)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::memcpy(target.data +
                    static_cast<std::size_t>(base) * target.footprint.stride,
                source.data, active * planned->source.identity.element_bytes);
    wrote = true;
  }
  return Status::success();
}

} // namespace rund::compute::detail
