#include "../projection/internal.hpp"

#include <algorithm>

namespace rund::compute::detail::graph_reduce {

bool graph_resource_index(const residency::TiledGraphPlan &plan,
                          const std::uint32_t resource,
                          std::size_t &index) noexcept {
  const auto found =
      std::find_if(plan.resources().begin(), plan.resources().end(),
                   [resource](const residency::TiledGraphResource &declared) {
                     return declared.resource == resource;
                   });
  if (found == plan.resources().end()) {
    return false;
  }
  index = static_cast<std::size_t>(found - plan.resources().begin());
  return index < residency::TiledGraphResourceCapacity;
}

bool capture_stage_effects(const residency::TiledGraphPlan &plan,
                           const residency::EpochLease lease,
                           StageEffects &effects) noexcept {
  effects = {};
  if (lease.ports.empty() || lease.bindings.empty()) {
    return false;
  }
  for (const residency::GraphLeasePort &port : lease.ports) {
    std::size_t resource_index = 0u;
    if (!graph_resource_index(plan, port.resource, resource_index) ||
        port.binding_count == 0u || port.binding_count > PipelineLeafCapacity ||
        port.first_binding > lease.bindings.size() ||
        port.binding_count > lease.bindings.size() - port.first_binding ||
        port.first_remap > lease.remaps.size() ||
        port.remap_count > lease.remaps.size() - port.first_remap) {
      return false;
    }
    const residency::TiledGraphResource &resource =
        plan.resources()[resource_index];
    if (port.remap_count != resource.remaps.size() ||
        !std::equal(resource.remaps.begin(), resource.remaps.end(),
                    lease.remaps.begin() +
                        static_cast<std::ptrdiff_t>(port.first_remap))) {
      return false;
    }
    if (residency::writes(port.access)) {
      if (effects.written[resource_index]) {
        return false;
      }
      effects.written[resource_index] = true;
      effects.regions[resource_index] = port.region;
      effects.counts[resource_index] = port.binding_count;
      for (std::size_t page = 0u; page < port.binding_count; ++page) {
        effects.keys[resource_index][page] =
            lease.bindings[port.first_binding + page].key;
      }
    } else {
      bool retires = true;
      for (std::size_t page = 0u; page < port.binding_count; ++page) {
        retires = lease.bindings[port.first_binding + page].retire_on_success &&
                  retires;
      }
      effects.retired[resource_index] = retires;
    }
  }
  return true;
}

void apply_stage_effects(Ticket &ticket, const StageEffects &effects) noexcept {
  for (std::size_t resource = 0u;
       resource < residency::TiledGraphResourceCapacity; ++resource) {
    if (effects.retired[resource]) {
      ticket.live_resources[resource] = {};
    }
    if (effects.written[resource]) {
      LiveResource &live = ticket.live_resources[resource];
      live.keys = effects.keys[resource];
      live.region = effects.regions[resource];
      live.count = effects.counts[resource];
      live.dirty = true;
    }
  }
}

} // namespace rund::compute::detail::graph_reduce
