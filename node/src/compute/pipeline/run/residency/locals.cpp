#include "../internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../state.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail {

Status residency_pipeline_locals(const std::shared_ptr<PipelineState> &state,
                                 const residency::EpochLease lease,
                                 const std::span<std::uint32_t> local_order,
                                 std::size_t &issued_steps) noexcept {
  const bool graph_lease = !lease.ports.empty();
  issued_steps = 0u;
  if (!valid_pipeline(state) || state->residency == nullptr ||
      lease.token == 0u || lease.bindings.empty() ||
      !state->publications.empty() || !state->windows.empty() ||
      local_order.size() < PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::Pool *const pool = state->residency_pool.get();
  if (pool == nullptr || state->residency_bank >= residency::Pool::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<bool, PipelineLeafCapacity> local_used{};
  if (graph_lease) {
    if (state->residency_graph_stage == residency::NoGraphStage ||
        state->residency_port_count == 0u ||
        state->residency_port_count != lease.ports.size() ||
        state->residency_port_count > state->residency_ports.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    issued_steps = lease.ports.front().binding_count;
    if (issued_steps == 0u || issued_steps > pool->layout.frame_capacity ||
        issued_steps > state->steps.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::size_t first_binding = 0u;
    std::size_t first_remap = 0u;
    for (std::size_t port_index = 0u; port_index < state->residency_port_count;
         ++port_index) {
      const PipelineResidencyPort &sealed = state->residency_ports[port_index];
      const residency::GraphLeasePort &issued = lease.ports[port_index];
      const residency::TiledGraphResource *const graph_resource =
          state->residency != nullptr && state->residency->graph_tiled()
              ? state->residency->tiled_graph().resource(sealed.resource)
              : nullptr;
      const residency::PoolPhysicalOwner *const physical_owner =
          graph_resource == nullptr
              ? nullptr
              : pool->graph_owner(graph_resource->physical_id);
      if (issued.program_port != sealed.program_port ||
          issued.access != sealed.access ||
          issued.resource != sealed.resource ||
          issued.region != sealed.region || physical_owner == nullptr ||
          physical_owner->arena == nullptr ||
          issued.cache_region_count != residency::Pool::BankCount ||
          issued.cache_regions != physical_owner->cache_regions ||
          issued.first_binding != first_binding ||
          issued.binding_count != issued_steps ||
          issued.first_remap != first_remap ||
          issued.remap_count != sealed.remap_count ||
          issued.first_remap > lease.remaps.size() ||
          issued.remap_count > lease.remaps.size() - issued.first_remap ||
          first_binding > lease.bindings.size() ||
          issued_steps > lease.bindings.size() - first_binding) {
        return Status::fail(Reason::PipelineInvalid);
      }
      for (std::size_t remap = 0u; remap < sealed.remap_count; ++remap) {
        if (lease.remaps[issued.first_remap + remap] != sealed.remaps[remap]) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      std::uint64_t base_page = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t page = 0u; page < issued_steps; ++page) {
        base_page =
            std::min(base_page, lease.bindings[first_binding + page].key.page);
      }
      for (std::size_t page = 0u; page < issued_steps; ++page) {
        const residency::CacheBinding &binding =
            lease.bindings[first_binding + page];
        if (binding.access != sealed.access ||
            binding.frame < sealed.region.first ||
            binding.frame >= sealed.region.first + sealed.region.count) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const std::size_t local = binding.frame - sealed.region.first;
        if (local >= pool->layout.frame_capacity ||
            local >= state->steps.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        std::size_t source = page;
        if (sealed.remap_count != 0u) {
          const auto found =
              std::find_if(sealed.remaps.begin(),
                           sealed.remaps.begin() +
                               static_cast<std::ptrdiff_t>(sealed.remap_count),
                           [page](const residency::GraphPageRemap remap) {
                             return remap.target_local == page;
                           });
          if (found == sealed.remaps.begin() +
                           static_cast<std::ptrdiff_t>(sealed.remap_count) ||
              found->source_local >= issued_steps) {
            return Status::fail(Reason::PipelineInvalid);
          }
          source = found->source_origin == residency::GraphPageOrigin::Begin
                       ? found->source_local
                       : issued_steps - 1u - found->source_local;
        }
        std::uint64_t expected_page = 0u;
        if (!kernel::checked::add(base_page, source, expected_page) ||
            binding.key.page != expected_page) {
          return Status::fail(Reason::PipelineInvalid);
        }
        if (port_index == 0u) {
          if (local_used[local]) {
            return Status::fail(Reason::PipelineInvalid);
          }
          local_used[local] = true;
          local_order[page] = static_cast<std::uint32_t>(local);
        } else {
          const residency::CacheBinding &anchor = lease.bindings[page];
          if (sealed.remap_count == 0u && binding.key.page != anchor.key.page) {
            return Status::fail(Reason::PipelineInvalid);
          }
          if (local != local_order[page]) {
            return Status::fail(Reason::PipelineInvalid);
          }
        }
      }
      first_binding += issued_steps;
      first_remap += issued.remap_count;
    }
    if (first_binding != lease.bindings.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    issued_steps = lease.bindings.size() / 2u;
    if (state->residency_stage != PipelineResidencyStage::Direct ||
        state->residency_graph_stage != residency::NoGraphStage ||
        state->residency_port_count != 0u || issued_steps == 0u ||
        lease.bindings.size() != issued_steps * 2u ||
        issued_steps > pool->layout.frame_capacity ||
        issued_steps > state->steps.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const residency::FrameRegion source_region =
        pool->input_regions[state->residency_bank];
    const residency::FrameRegion destination_region{
        .tier = source_region.tier,
        .role = residency::FrameRole::Output,
        .first = pool->first_output_frame +
                 state->residency_bank * pool->layout.frame_capacity,
        .count = pool->layout.frame_capacity,
    };
    for (std::size_t index = 0u; index < issued_steps; ++index) {
      const residency::CacheBinding &source = lease.bindings[index];
      const residency::CacheBinding &destination =
          lease.bindings[issued_steps + index];
      if (source.access != residency::Access::Read ||
          destination.access != residency::Access::Write ||
          source.key.page != destination.key.page ||
          source.frame < source_region.first ||
          source.frame >= source_region.first + source_region.count ||
          destination.frame < destination_region.first ||
          destination.frame >=
              destination_region.first + destination_region.count) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::size_t source_local = source.frame - source_region.first;
      const std::size_t destination_local =
          destination.frame - destination_region.first;
      if (source_local >= state->steps.size() ||
          destination_local != source_local || local_used[source_local]) {
        return Status::fail(Reason::PipelineInvalid);
      }
      local_used[source_local] = true;
      local_order[index] = static_cast<std::uint32_t>(source_local);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail
