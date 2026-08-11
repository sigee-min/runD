#include "local.hpp"

#include "../arena.hpp"
#include "../compare.hpp"
#include "../prepare.hpp"
#include "../resource.hpp"

#include "../../../../accel/kernel/recurrence.hpp"
#include "../../../backend.hpp"
#include "../../../buffer/local.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../job/local.hpp"
#include "../../../memory/arena.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Status materialize_pipeline(PipelineBuildState &build) {
  try {
    if (build.memory == nullptr || build.memory->frozen == nullptr ||
        build.memory->resources.size() !=
            build.memory->hazards.lifetimes.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineBuildSnapshot &frozen = *build.memory->frozen;
    std::array<std::uint32_t, PipelineIterationCapacity> ordinals{};
    build.materialized_resources.clear();
    build.materialized_resources.reserve(build.memory->resources.size());
    for (const PipelineResolvedResourcePlan &resource :
         build.memory->resources) {
      std::shared_ptr<BufferState> owner;
      bool external_owner = false;
      if (const auto *external =
              std::get_if<PipelineExternalResourcePlan>(&resource.locator)) {
        owner = external->owner;
        external_owner = true;
      } else {
        const auto *internal =
            std::get_if<PipelineInternalResourcePlan>(&resource.locator);
        if (internal == nullptr ||
            resource.count > std::numeric_limits<std::size_t>::max()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        if (internal->owner != nullptr) {
          owner = internal->owner;
        } else {
          auto made = make_planned_input_binding_buffer(
              build.device, resource.type,
              static_cast<std::size_t>(resource.count),
              resource.physical_bytes);
          if (!made) {
            return Status::fail(made.reason());
          }
          owner = std::move(made).value();
        }
        if (internal->fill == PipelineFill::Ordinal) {
          if (resource.type != Type::U32 || resource.count > ordinals.size()) {
            return Status::fail(Reason::PipelineInvalid);
          }
          const auto count = static_cast<std::size_t>(resource.count);
          std::iota(ordinals.begin(), ordinals.begin() + count, 0u);
          WriteStats writes{};
          const Status written = write_buffer(owner,
                                              HostView{.data = ordinals.data(),
                                                       .count = count,
                                                       .type = Type::U32},
                                              writes);
          if (!written) {
            return written;
          }
        }
      }
      if (owner == nullptr) {
        return Status::fail(Reason::BindingInvalid);
      }
      // External semantic validation remains prepare admission's authority so
      // it preserves the public BindingDeviceMismatch/type/shape reasons.
      // Materialization only retains that already-sealed owner. Internals are
      // created here and therefore must exactly match this constructor's plan.
      if (!external_owner &&
          (owner->device != frozen.device || owner->type != resource.type ||
           owner->count != resource.count || owner->bytes != resource.bytes ||
           owner->physical_bytes != resource.physical_bytes)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      build.materialized_resources.push_back(owner);
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

[[nodiscard]] Result<PipelineMemorySet>
make_pipeline_memory(const std::shared_ptr<DeviceState> &device,
                     const std::span<const PipelineFrozenStep> steps,
                     const PipelineMemoryPlan &plan) {
  try {
    PipelineMemorySet result;
    result.steps.resize(steps.size());
    if (plan.cpu_storage_by_step.size() != steps.size() ||
        plan.workspace_routes.size() != steps.size() ||
        plan.cpu_route_slices.size() != steps.size() ||
        (device->backend == Backend::Cpu &&
         (plan.cpu_job_slices.size() != steps.size() ||
          plan.cpu_workspace_slices.size() != steps.size())) ||
        (!plan.cpu_alternate_job_slices.empty() &&
         plan.cpu_alternate_job_slices.size() != steps.size()) ||
        (!plan.cpu_alternate_route_slices.empty() &&
         plan.cpu_alternate_route_slices.size() != steps.size())) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.cpu_storage_by_step = plan.cpu_storage_by_step;
    if (plan.cpu_storage_plans.size() != plan.cpu_programs.size() ||
        plan.cpu_route_plans.size() != plan.cpu_programs.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    const bool has_cpu_prepared_jobs =
        device->backend == Backend::Cpu && !steps.empty();
    if (has_cpu_prepared_jobs != plan.cpu_prepared_arena.layout.sealed) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    if (has_cpu_prepared_jobs) {
      auto prepared = make_cpu_prepared_arena(plan.cpu_prepared_arena);
      if (!prepared) {
        return Result<PipelineMemorySet>::fail(prepared.reason());
      }
      result.cpu_prepared_arena = std::move(prepared).value();
    }
    result.cpu_storage.reserve(plan.cpu_programs.size());
    for (std::size_t index = 0u; index < plan.cpu_programs.size(); ++index) {
      auto storage = make_cpu_graph_storage(plan.cpu_programs[index],
                                            plan.cpu_storage_plans[index],
                                            result.cpu_prepared_arena);
      if (!storage || storage.value() == nullptr) {
        return Result<PipelineMemorySet>::fail(
            storage ? Reason::CpuRuntimeInvalid : storage.reason());
      }
      result.cpu_storage.push_back(std::move(storage).value());
    }
    if (result.cpu_prepared_arena != nullptr &&
        !result.cpu_prepared_arena->claims_complete(
            plan.cpu_prepared_arena.execution)) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    if (plan.steps.size() != steps.size() + 1u || plan.steps.empty() ||
        plan.steps.front() != 0u || plan.steps.back() != plan.offsets.size() ||
        plan.owners.size() != plan.offsets.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.buffers.reserve(plan.chunks.size());
    for (const std::size_t count : plan.chunks) {
      if (count == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_planned_workspace_buffer(device, count);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      const Status committed = validate_buffer_commitment(*device, **made);
      if (!committed) {
        return Result<PipelineMemorySet>::fail(committed.reason());
      }
      result.buffers.push_back(std::move(made).value());
    }
    if (plan.views.size() != steps.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.prepared.reserve(plan.view_chunks.size() + plan.scratch.size());
    if (!plan.view_chunks.empty() || !plan.scratch.empty()) {
      result.arena = std::make_shared<JobArena>();
      result.arena->buffers.reserve(plan.view_chunks.size() +
                                    plan.scratch.size());
      result.arena->slots.reserve(plan.view_slots.size() + plan.scratch.size());
      result.arena->scratch.reserve(plan.scratch.size());
    }
    for (const std::size_t words : plan.view_chunks) {
      if (words == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_planned_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      const Status committed = validate_buffer_commitment(*device, **made);
      if (!committed) {
        return Result<PipelineMemorySet>::fail(committed.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
    }
    for (const PipelineMemoryPlan::ViewSlot slot : plan.view_slots) {
      if (slot.words == 0u || slot.owner >= plan.view_chunks.size() ||
          slot.offset_words > plan.view_chunks[slot.owner] ||
          slot.words > plan.view_chunks[slot.owner] - slot.offset_words) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      result.arena->slots.push_back(JobArenaSlot{
          .words = slot.words,
          .owner = slot.owner,
          .offset_words = slot.offset_words,
      });
    }
    for (std::size_t index = 0u; index < plan.scratch.size(); ++index) {
      const node::accel::detail::KernelScratchPage scratch =
          plan.scratch[index];
      if (scratch.bytes == 0u || scratch.bytes % memory::Word != 0u ||
          scratch.bytes / memory::Word >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max()) ||
          scratch.slot != plan.view_slots.size() + index ||
          result.arena == nullptr) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      const std::size_t words =
          static_cast<std::size_t>(scratch.bytes / memory::Word);
      auto made = make_planned_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      const Status committed = validate_buffer_commitment(*device, **made);
      if (!committed) {
        return Result<PipelineMemorySet>::fail(committed.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
      const std::size_t slot = result.arena->slots.size();
      if (slot != scratch.slot) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      result.arena->slots.push_back(
          JobArenaSlot{.words = words,
                       .owner = result.arena->buffers.size() - 1u,
                       .offset_words = 0u});
      result.arena->scratch.push_back(scratch);
    }
    for (std::size_t step_index = 0u; step_index < steps.size(); ++step_index) {
      const PipelineFrozenStep &step = steps[step_index];
      const auto &chunks = step.program->chunks;
      const PipelineWorkspaceRoute workspace_route =
          plan.workspace_routes[step_index];
      if (!workspace_route.present()) {
        continue;
      }
      const std::size_t workspace_owner = workspace_route.owner;
      if (workspace_owner != step_index) {
        if (workspace_owner >= step_index ||
            !plan.workspace_routes[workspace_owner].owns(workspace_owner) ||
            result.steps[workspace_owner] == nullptr ||
            result.steps[workspace_owner]->program != step.program) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        result.steps[step_index] = result.steps[workspace_owner];
        continue;
      }
      const std::vector<std::uint32_t> &order = step.program->chunk_order;
      if (order.size() != chunks.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      std::shared_ptr<JobWorkspace> workspace;
      if (device->backend == Backend::Cpu) {
        if (result.cpu_prepared_arena == nullptr) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        CpuWorkspaceStorage storage{};
        if (!result.cpu_prepared_arena->view(
                plan.cpu_workspace_slices[step_index], storage) ||
            storage.workspace == nullptr ||
            !storage.workspace->buffers.bind(storage.buffers, chunks.size()) ||
            !storage.workspace->offsets.bind(storage.offsets, chunks.size())) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        workspace = std::shared_ptr<JobWorkspace>(result.cpu_prepared_arena,
                                                  storage.workspace);
      } else {
        workspace = std::make_shared<JobWorkspace>();
        workspace->buffers.resize(chunks.size());
        workspace->offsets.resize(chunks.size());
      }
      workspace->program = step.program;
      workspace->arena = result.arena;
      const std::size_t begin = plan.steps[step_index];
      const std::size_t end = plan.steps[step_index + 1u];
      if (begin > end || end > plan.offsets.size() ||
          end - begin != order.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      for (std::size_t rank = 0u; rank < order.size(); ++rank) {
        const std::size_t chunk = order[rank];
        const std::size_t owner = plan.owners[begin + rank];
        const std::size_t offset = plan.offsets[begin + rank];
        if (chunk >= chunks.size() || owner >= result.buffers.size() ||
            owner >= plan.chunks.size() || offset > plan.chunks[owner] ||
            chunks[chunk].count > plan.chunks[owner] - offset) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        workspace->buffers[chunk] = result.buffers[owner];
        workspace->offsets[chunk] = offset;
      }
      result.steps[step_index] = std::move(workspace);
    }
    return Result<PipelineMemorySet>::success(std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<PipelineMemorySet>::fail(Reason::BufferCapacity);
  }
}
} // namespace rund::compute::detail
