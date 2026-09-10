#include "backing_set.hpp"

#include "../backing.hpp"
#include "cache.hpp"
#include "projection.hpp"

#include "../../device/residency/pool.hpp"
#include "../../device/residency/registry/graph_persist_owner.hpp"
#include "../../device/residency/registry/view_owner.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <span>

namespace rund::compute::detail {

Status lock_pool(VirtualPipelineState &state,
                 VirtualRunResources &resources) noexcept {
  if (resources.view_commit != nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (resources.pool.owns_lock()) {
    return Status::success();
  }
  if (state.pipeline == nullptr || state.pipeline->residency_pool == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  residency::Pool &pool = *state.pipeline->residency_pool;
  resources.pool =
      std::unique_lock<std::mutex>{pool.execution_gate(), std::try_to_lock};
  return resources.pool.owns_lock() ? Status::success()
                                    : Status::fail(Reason::PipelineBusy);
}

bool lock_backings(VirtualPipelineState &state,
                   VirtualRunResources &resources) noexcept {
  resources.backings = {};
  resources.pool = std::unique_lock<std::mutex>{};
  resources.page_count = 0u;
  if (state.input_count == 0u ||
      state.input_count > VirtualPipelineState::InputCapacity ||
      state.output == nullptr || state.output->backing == nullptr) {
    return false;
  }
  std::array<std::mutex *, VirtualPipelineState::InputCapacity + 1u> gates{};
  for (std::size_t index = 0u; index < state.input_count; ++index) {
    const VirtualBufferState *const input = virtual_input(state, index);
    if (input == nullptr || input->backing == nullptr) {
      return false;
    }
    gates[index] = &VirtualBackingAccess::gate(*input->backing);
  }
  gates[state.input_count] =
      &VirtualBackingAccess::gate(*state.output->backing);
  resources.backings.count = state.input_count + 1u;
  std::sort(gates.begin(), gates.begin() + resources.backings.count,
            std::less<std::mutex *>{});
  if (std::adjacent_find(gates.begin(),
                         gates.begin() + resources.backings.count) !=
      gates.begin() + resources.backings.count) {
    resources.backings = {};
    return false;
  }
  for (std::size_t index = 0u; index < resources.backings.count; ++index) {
    resources.backings.locks[index] =
        std::unique_lock<std::mutex>{*gates[index]};
  }
  return true;
}

namespace {

[[nodiscard]] residency::Authority *
receipt_authority(VirtualRunResources &resources) noexcept {
  return resources.view_commit == nullptr ? nullptr
                                          : resources.view_commit->authority();
}

[[nodiscard]] Status quarantine_stage(VirtualRunResources &resources) noexcept {
  residency::Authority *const authority = receipt_authority(resources);
  if (authority == nullptr ||
      !authority->views().quarantine_view_commit(resources.view_commit) ||
      resources.view_commit != nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::fail(Reason::DeviceLost);
}

} // namespace

Status close_pool_stage(VirtualRunResources &resources) noexcept {
  if (resources.view_commit == nullptr) {
    return Status::success();
  }
  residency::Authority *const authority = receipt_authority(resources);
  if (authority != nullptr &&
      authority->views().close_view_commit(resources.view_commit)) {
    return Status::success();
  }
  if (resources.view_commit == nullptr) {
    return Status::fail(Reason::DeviceLost);
  }
  return quarantine_stage(resources);
}

Status quarantine_pool_stage(VirtualRunResources &resources) noexcept {
  if (resources.view_commit == nullptr) {
    return Status::fail(Reason::DeviceLost);
  }
  return quarantine_stage(resources);
}

Status stage_pool(VirtualPipelineState &state, const VirtualRunProjection &run,
                  Stats &stats, VirtualRunResources &resources,
                  bool &poison) noexcept {
  if (resources.view_commit != nullptr) {
    poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state.pipeline == nullptr || state.pipeline->residency_pool == nullptr) {
    poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }

  residency::Pool &pool = *state.pipeline->residency_pool;
  auto persist_owner = pool.authority().graph_persists();
  if (!resources.pool.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }

  resources.page_count = run.graph_execution() ? run.active.graph.page_count()
                                               : run.active.stream.page_count();
  const bool retry =
      run.graph_execution() && state.pipeline->device != nullptr &&
      state.pipeline->device->backend == Backend::Cpu &&
      state.cpu_receipts != nullptr &&
      persist_owner.has_cpu_graph_retry(state.cpu_receipts->domain());

  std::array<residency::FrameRegion,
             residency::TiledGraphResourceCapacity * residency::Pool::BankCount>
      views{};
  std::size_t view_count = 0u;
  if (run.graph_execution() && !retry) {
    for (const residency::PoolPhysicalOwner &owner : pool.graph_owners) {
      for (const residency::FrameRegion region : owner.cache_regions) {
        if (view_count >= views.size()) {
          poison = true;
          return Status::fail(Reason::PipelineInvalid);
        }
        views[view_count++] = region;
      }
    }
  }

  if (resources.page_count == 0u) {
    if (run.graph_execution() && !retry) {
      const residency::registry_model::ViewCommitPlan plan{
          .kind = residency::registry_model::ViewCommitPlan::Kind::Graph,
          .activation_regions =
              std::span<const residency::FrameRegion>{views.data(), view_count},
          .graph = &run.active.graph,
      };
      const residency::ViewActivationResult committed =
          pool.authority().views().commit_view_plan(plan,
                                                    &resources.view_commit);
      if (!committed || resources.view_commit == nullptr) {
        poison = committed.failure == residency::AuthorityFailure::Invalid;
        return Status::fail(committed.failure ==
                                    residency::AuthorityFailure::Busy
                                ? Reason::PipelineBusy
                                : Reason::PipelineInvalid);
      }
      ::rund::detail::counter::Accumulate(
          stats.pipeline.residency.eviction_count, committed.eviction_count);
    }
    return Status::success();
  }
  const std::uint64_t last_page = resources.page_count - 1u;
  residency::CacheKey last{};
  const bool projected =
      run.graph_execution()
          ? residency::project_graph_cache_key(
                run.graph_input,
                residency::PageKey{.resource = run.graph_input.resource,
                                   .page = last_page},
                last)
          : (last = virtual_cache_key(run, run.input_backing, run.input_version,
                                      last_page),
             true);
  bool execution_frozen = false;
  bool host_frozen = false;
  if (retry) {
    execution_frozen = projected;
    host_frozen = projected;
  } else if (projected && run.graph_execution()) {
    const residency::TiledGraphPlan *const graph =
        state.pipeline->residency != nullptr &&
                state.pipeline->residency->graph_tiled()
            ? &state.pipeline->residency->tiled_graph()
            : nullptr;
    std::array<std::array<residency::FrameRegion,
                          residency::GraphFreezeRequest::RegionCapacity>,
               residency::registry_model::ViewCommitPlan::GraphRequestCapacity>
        regions{};
    std::array<residency::GraphFreezeRequest,
               residency::registry_model::ViewCommitPlan::GraphRequestCapacity>
        requests{};
    bool complete = graph != nullptr && run.input_count != 0u &&
                    run.input_count == state.graph_input_resource_count;
    for (std::size_t index = 0u; complete && index < run.input_count; ++index) {
      const std::uint32_t resource_id = state.graph_input_resources[index];
      const residency::TiledGraphResource *const input_resource =
          graph->resource(resource_id);
      const residency::PoolPhysicalOwner *const input_owner =
          input_resource == nullptr
              ? nullptr
              : pool.graph_owner(input_resource->physical_id);
      const auto materialization = std::find_if(
          run.graph_materializations.begin(),
          run.graph_materializations.begin() + run.graph_resource_count,
          [resource_id](const residency::GraphMaterialization &candidate) {
            return candidate.resource == resource_id;
          });
      if (input_owner == nullptr || input_owner->arena == nullptr ||
          materialization ==
              run.graph_materializations.begin() + run.graph_resource_count) {
        complete = false;
        break;
      }
      std::copy(input_owner->cache_regions.begin(),
                input_owner->cache_regions.end(), regions[index].begin());
      std::size_t count = residency::Pool::BankCount;
      complete = count <= residency::GraphFreezeRequest::RegionCapacity;
      if (pool.layout.graph_host_service) {
        for (std::uint32_t bank = 0u;
             complete && bank < residency::Pool::BankCount; ++bank) {
          const residency::FrameRegion host =
              virtual_graph_host_input_region(run, index, bank);
          complete = host.count == run.host_frame_capacity &&
                     count < residency::GraphFreezeRequest::RegionCapacity;
          if (complete) {
            regions[index][count++] = host;
          }
        }
      }
      requests[index] = residency::GraphFreezeRequest{
          .materialization = *materialization,
          .regions =
              std::span<const residency::FrameRegion>{regions[index].data(),
                                                      count},
      };
    }
    if (complete) {
      const residency::registry_model::ViewCommitPlan plan{
          .kind = residency::registry_model::ViewCommitPlan::Kind::Graph,
          .activation_regions =
              std::span<const residency::FrameRegion>{views.data(), view_count},
          .graph = &run.active.graph,
          .requests =
              std::span<const residency::GraphFreezeRequest>{requests.data(),
                                                             run.input_count},
      };
      const residency::ViewActivationResult committed =
          pool.authority().views().commit_view_plan(plan,
                                                    &resources.view_commit);
      if (committed && resources.view_commit != nullptr) {
        execution_frozen = true;
        host_frozen = true;
        ::rund::detail::counter::Accumulate(
            stats.pipeline.residency.eviction_count, committed.eviction_count);
      } else {
        poison = committed.failure == residency::AuthorityFailure::Invalid;
        return Status::fail(committed.failure ==
                                    residency::AuthorityFailure::Busy
                                ? Reason::PipelineBusy
                                : Reason::PipelineMemoryBudget);
      }
    }
  } else if (projected) {
    std::array<residency::FrameRegion, residency::Pool::BankCount + 1u>
        regions{};
    std::copy(pool.input_regions.begin(), pool.input_regions.end(),
              regions.begin());
    std::size_t count = residency::Pool::BankCount;
    if (pool.host_input_frame_count != 0u) {
      regions[count++] = residency::FrameRegion{
          .tier = residency::FrameTier::Host,
          .role = residency::FrameRole::Input,
          .first = pool.first_host_input_frame,
          .count = pool.host_input_frame_count,
      };
    }
    const residency::registry_model::ViewCommitPlan plan{
        .kind = residency::registry_model::ViewCommitPlan::Kind::Stream,
        .stream_regions =
            std::span<const residency::FrameRegion>{regions.data(), count},
        .stream = &run.active.stream,
        .last = last,
    };
    const residency::ViewActivationResult committed =
        pool.authority().views().commit_view_plan(plan, &resources.view_commit);
    if (committed && resources.view_commit != nullptr) {
      execution_frozen = true;
      host_frozen = true;
      ::rund::detail::counter::Accumulate(
          stats.pipeline.residency.eviction_count, committed.eviction_count);
    } else {
      poison = committed.failure == residency::AuthorityFailure::Invalid;
      return Status::fail(committed.failure == residency::AuthorityFailure::Busy
                              ? Reason::PipelineBusy
                              : Reason::PipelineMemoryBudget);
    }
  }
  if (!execution_frozen || !host_frozen) {
    poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }
  stats.pipeline.residency.resident_frames_peak = std::min(
      resources.page_count, run.frame_capacity * residency::Pool::BankCount);
  return Status::success();
}

} // namespace rund::compute::detail
