#include "../model.hpp"

#include "../../../type.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail::residency {
namespace {


[[nodiscard]] bool remap_source(const TiledGraphResource &resource,
                                const std::size_t target,
                                const std::uint64_t frame_capacity,
                                const std::uint64_t active_count,
                                std::uint64_t &source) noexcept {
  if (resource.remaps.empty()) {
    source = target;
    return true;
  }
  if (resource.remaps.size() != frame_capacity || target >= active_count) {
    return false;
  }
  if (target >= resource.remaps.size()) {
    return false;
  }
  // The planner seals a complete bijection sorted by target_local.
  const GraphPageRemap &remap = resource.remaps[target];
  if (remap.target_local != target || remap.source_local >= active_count) {
    return false;
  }
  source = remap.source_origin == GraphPageOrigin::Begin
               ? remap.source_local
               : active_count - 1u - remap.source_local;
  return true;
}

} // namespace

bool TiledGraphInvocation::project(const std::uint64_t batch_index,
                                   const std::size_t stage_index,
                                   const std::span<PageUse> uses,
                                   Epoch &epoch) const noexcept {
  PageRun run{};
  std::uint64_t ordinal = 0u;
  if (!valid() || stage_index >= plan_->stages_.size() ||
      batch_index > std::numeric_limits<std::uint32_t>::max() ||
      !batch(batch_index, run) || plan_->stages_[stage_index].ports.empty() ||
      run.page_count > std::numeric_limits<std::size_t>::max() /
                           plan_->stages_[stage_index].ports.size() ||
      uses.size() !=
          run.page_count * plan_->stages_[stage_index].ports.size() ||
      !kernel::checked::mul(batch_index, plan_->stage_count(), ordinal) ||
      !kernel::checked::add(ordinal, stage_index, ordinal)) {
    return false;
  }
  const TiledGraphStage &stage = plan_->stages_[stage_index];
  const std::uint64_t batch_base = ordinal - stage_index;
  std::uint64_t prefetch_span = 0u;
  if (!kernel::checked::mul(prefetch_distance(), plan_->stage_count(),
                            prefetch_span)) {
    return false;
  }
  for (std::size_t port_index = 0u; port_index < stage.ports.size();
       ++port_index) {
    const TiledGraphPort port = stage.ports[port_index];
    const TiledGraphResource *const resource = plan_->resource(port.resource);
    if (resource == nullptr ||
        (stage.domain != StageDomain::Tile &&
         stage.domain != StageDomain::TilePartial) ||
        resource->first_stage == NoGraphStage ||
        resource->last_stage == NoGraphStage ||
        resource->first_stage > resource->last_stage ||
        resource->last_stage >= plan_->stage_count() ||
        (port.next_stage != NoGraphStage &&
         (port.next_stage <= stage_index ||
          port.next_stage >= plan_->stage_count()))) {
      return false;
    }
    const std::size_t resource_index =
        static_cast<std::size_t>(resource - plan_->resources_.data());
    std::uint64_t next_use = NeverUse;
    std::uint64_t first_pin = 0u;
    std::uint64_t last_pin = 0u;
    if (!kernel::checked::add(batch_base, resource->first_stage, first_pin) ||
        !kernel::checked::add(batch_base, resource->last_stage, last_pin)) {
      return false;
    }
    // Backing data is rematerializable, so its hard pin is only the exact
    // consumer epoch; next_use independently expresses profitable cache
    // retention. Transient values and final publication have no backing
    // source and retain their producer-to-last-consumer closed interval.
    if (resource->persistence == ResourcePersistence::Backing &&
        reads(port.access)) {
      first_pin = ordinal;
      last_pin = ordinal;
    }
    if (port.next_stage != NoGraphStage) {
      if (!kernel::checked::add(ordinal,
                                static_cast<std::uint64_t>(port.next_stage) -
                                    static_cast<std::uint64_t>(stage_index),
                                next_use)) {
        next_use = NeverUse;
      }
    } else if (resource->persistence == ResourcePersistence::Backing &&
               reads(port.access)) {
      std::uint64_t recurrent = 0u;
      if (!kernel::checked::add(epoch_count(), first_pin, recurrent)) {
        recurrent = NeverUse;
      }
      next_use = recurrent;
    }
    const std::uint64_t prefetch_epoch =
        resource->persistence == ResourcePersistence::Backing &&
                reads(port.access)
            ? (ordinal > prefetch_span ? ordinal - prefetch_span : 0u)
            : ordinal;
    for (std::size_t index = 0u; index < run.page_count; ++index) {
      std::uint64_t source_index = 0u;
      std::uint64_t page = 0u;
      std::uint64_t logical_offset = 0u;
      if (!remap_source(*resource, index, frame_capacity(), run.page_count,
                        source_index) ||
          !kernel::checked::add(run.first_page, source_index, page) ||
          !kernel::checked::mul(page, resource->page_bytes, logical_offset) ||
          logical_offset >= logical_bytes_[resource_index]) {
        return false;
      }
      uses[port_index * static_cast<std::size_t>(run.page_count) + index] =
          PageUse{
              .key = PageKey{.resource = resource->resource, .page = page},
              .access = port.access,
              .dirty = writes(port.access)
                           ? DirtyRange{.bytes = std::min(
                                            resource->page_bytes,
                                            logical_bytes_[resource_index] -
                                                logical_offset)}
                           : DirtyRange{},
              .next_use = next_use,
              .pin =
                  PinInterval{.first_epoch = first_pin, .last_epoch = last_pin},
              .prefetch_epoch = prefetch_epoch,
              .ready_epoch = ordinal,
          };
    }
  }
  epoch = Epoch{
      .ordinal = ordinal,
      .node = stage.node,
      .tile = static_cast<std::uint32_t>(batch_index),
      .first_use = 0u,
      .use_count = uses.size(),
  };
  return true;
}

} // namespace rund::compute::detail::residency
