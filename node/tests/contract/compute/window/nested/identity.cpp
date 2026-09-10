#include "identity.hpp"
#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {

[[nodiscard]] constexpr JobViewIdentity
BindingView(const rund::compute::detail::JobBufferView view) noexcept {
  return JobViewIdentity{.offset = view.offset,
                         .count = view.count,
                         .stride = view.stride,
                         .element_bytes = view.element_bytes,
                         .alignment = view.alignment};
}

template <class Owners>
void CaptureBufferOwners(
    const Owners &source,
    std::vector<const rund::compute::detail::BufferState *> &target,
    bool &valid) {
  target.reserve(source.size());
  for (const auto &owner : source) {
    target.push_back(owner.get());
    valid = valid && owner != nullptr;
  }
}

[[nodiscard]] JobBindingIdentity
CaptureJobBinding(const rund::compute::detail::JobState &job,
                  const rund::compute::Backend backend, bool &valid) {
  using namespace rund::compute::detail;
  JobBindingIdentity result{};
  result.owner = &job;
  result.program = job.program.get();
  result.program_fingerprint = job.program == nullptr
                                   ? rund::compute::graph::Fingerprint{}
                                   : job.program->graph_info.fingerprint;
  result.workspace = job.workspace.get();
  result.workspace_program =
      job.workspace == nullptr ? nullptr : job.workspace->program.get();
  result.arena =
      job.workspace == nullptr ? nullptr : job.workspace->arena.get();
  result.prepared_owner = job.prepared.owner.get();
  result.write_prepared_owner = job.write_prepared.owner.get();
  result.prepared_ok = job.prepared.ok;
  result.write_prepared_ok = job.write_prepared.ok;
  valid = valid && job.program != nullptr &&
          (job.workspace == nullptr || job.workspace->program == job.program);
  if (backend != rund::compute::Backend::Cpu) {
    valid = valid && job.prepared.ok && job.prepared.owner != nullptr;
  }

  CaptureBufferOwners(job.inputs, result.inputs, valid);
  CaptureBufferOwners(job.write_inputs, result.write_inputs, valid);
  CaptureBufferOwners(job.graph_buffers, result.graph_buffers, valid);
  CaptureBufferOwners(job.outputs, result.outputs, valid);
  const auto graph = job_graph_buffers(job);
  result.effective_graph.reserve(graph.size());
  for (const auto &owner : graph) {
    result.effective_graph.push_back(owner.get());
    valid = valid && owner != nullptr;
  }
  result.input_views.reserve(job.input_views.size());
  for (const JobBufferView view : job.input_views) {
    result.input_views.push_back(BindingView(view));
  }
  result.output_views.reserve(job.output_views.size());
  for (const JobBufferView view : job.output_views) {
    result.output_views.push_back(BindingView(view));
  }
  valid = valid && result.input_views.size() == result.inputs.size() &&
          result.output_views.size() == result.outputs.size();
  result.kernel_views.reserve(job.views.size());
  for (const rund::node::accel::detail::KernelViewSlot view : job.views) {
    result.kernel_views.push_back(
        KernelViewIdentity{.binding = view.binding,
                           .slot = view.slot,
                           .backing_bytes = view.backing_bytes,
                           .offset_bytes = view.offset_bytes,
                           .count = view.count,
                           .stride_bytes = view.stride_bytes,
                           .element_bytes = view.element_bytes,
                           .usage = view.usage});
  }
  const auto capture_transfers = [&](const auto &source, const auto &owners,
                                     std::vector<CpuTransferIdentity> &target) {
    target.reserve(source.size());
    for (const CpuViewTransfer &transfer : source) {
      const BufferState *const staging = transfer.binding < owners.size()
                                             ? owners[transfer.binding].get()
                                             : nullptr;
      target.push_back(CpuTransferIdentity{.external = transfer.external.get(),
                                           .staging = staging,
                                           .view = BindingView(transfer.view),
                                           .binding = transfer.binding});
      valid = valid && transfer.external != nullptr && staging != nullptr;
    }
  };
  capture_transfers(job.cpu_view_inputs, job.inputs, result.cpu_inputs);
  capture_transfers(job.cpu_view_outputs, job.outputs, result.cpu_outputs);

  if (job.workspace != nullptr) {
    CaptureBufferOwners(job.workspace->buffers, result.workspace_buffers,
                        valid);
    result.workspace_offsets.assign(job.workspace->offsets.begin(),
                                    job.workspace->offsets.end());
    valid = valid &&
            result.workspace_buffers.size() == result.workspace_offsets.size();
  }
  if (result.arena != nullptr) {
    const JobArena &arena = *job.workspace->arena;
    result.arena_bound = arena.bound;
    result.arena_binds_heap = arena.binds.heap;
    result.arena_binds_ok = arena.binds.ok;
    CaptureBufferOwners(arena.buffers, result.arena_buffers, valid);
    result.arena_slots.reserve(arena.slots.size());
    for (const JobArenaSlot slot : arena.slots) {
      result.arena_slots.push_back(ArenaSlotIdentity{
          .words = slot.words,
          .owner = slot.owner,
          .offset_words = slot.offset_words,
      });
    }
    result.scratch.reserve(arena.scratch.size());
    for (const rund::node::accel::detail::KernelScratchPage page :
         arena.scratch) {
      result.scratch.push_back(
          ScratchIdentity{.slot = page.slot, .bytes = page.bytes});
    }
    valid = valid && arena.binds.valid();
    const auto *const refs = arena.binds.refs();
    const auto *const handles = arena.binds.handles();
    if (arena.binds.size() != 0u && (refs == nullptr || handles == nullptr)) {
      valid = false;
    } else {
      result.arena_bindings.reserve(
          static_cast<std::size_t>(arena.binds.size()));
      for (std::uint64_t index = 0u; index < arena.binds.size(); ++index) {
        const rund::kernel::ResidentBufferRef &ref = refs[index];
        result.arena_bindings.push_back(ResidentBindingIdentity{
            .handle = handles[index].get(),
            .id = ref.id,
            .bytes = ref.bytes,
            .offset_bytes = ref.offset_bytes,
            .element_bytes = ref.element_bytes,
            .stride_bytes = ref.stride_bytes,
            .count = ref.count,
            .usage = ref.usage,
        });
        valid = valid && handles[index] != nullptr;
      }
    }
  }
  return result;
}

[[nodiscard]] PipelineBindingIdentity
CaptureBindingIdentity(const rund::compute::Pipeline &pipeline,
                       const rund::compute::Backend backend) {
  using namespace rund::compute::detail;
  PipelineBindingIdentity result{};
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  result.owner = state.get();
  result.valid = state != nullptr && state->device != nullptr &&
                 state->device->backend == backend;
  if (state == nullptr) {
    return result;
  }
  result.prepared_owner = state->prepared.owner.get();
  result.alternate_prepared_owner = state->alternate_prepared.owner.get();
  result.prepared_ok = state->prepared.ok;
  result.alternate_prepared_ok = state->alternate_prepared.ok;
  result.transactional = state->transactional;
  if (backend != rund::compute::Backend::Cpu) {
    result.valid =
        result.valid && state->prepared.ok && state->prepared.owner != nullptr;
    if (state->transactional) {
      result.valid = result.valid && state->alternate_prepared.ok &&
                     state->alternate_prepared.owner != nullptr &&
                     state->prepared.owner != state->alternate_prepared.owner;
    } else {
      result.valid = result.valid && !state->alternate_prepared.ok &&
                     state->alternate_prepared.owner == nullptr;
    }
  }

  const auto append_unique = [](const JobState *const owner,
                                std::vector<const JobState *> &owners) {
    if (std::find(owners.begin(), owners.end(), owner) == owners.end()) {
      owners.push_back(owner);
    }
  };
  const auto capture_job = [&](const std::shared_ptr<JobState> &job) {
    if (job == nullptr) {
      result.valid = false;
      return;
    }
    const auto found =
        std::find_if(result.jobs.begin(), result.jobs.end(),
                     [&](const auto &id) { return id.owner == job.get(); });
    if (found == result.jobs.end()) {
      result.jobs.push_back(CaptureJobBinding(*job, backend, result.valid));
    }
  };
  result.steps.reserve(state->steps.size());
  for (const PipelineStep &step : state->steps) {
    result.steps.push_back(StepBindingIdentity{
        .program = step.program.get(),
        .normal = step.job.get(),
        .alternate = step.alternate_job.get(),
        .route = step.route,
        .logical_step = step.logical_step,
        .iteration = step.iteration,
        .window = step.window,
    });
    result.valid = result.valid && step.program != nullptr &&
                   step.job != nullptr && step.job->program == step.program;
    append_unique(step.job.get(), result.normal_jobs);
    capture_job(step.job);
    if (state->transactional) {
      result.valid = result.valid && step.alternate_job != nullptr &&
                     step.alternate_job != step.job &&
                     step.alternate_job->program == step.program &&
                     step.alternate_job->workspace == step.job->workspace;
      append_unique(step.alternate_job.get(), result.alternate_jobs);
      capture_job(step.alternate_job);
    } else if (step.alternate_job != nullptr) {
      result.valid = false;
    }
  }
  if (state->transactional) {
    result.valid = result.valid &&
                   result.normal_jobs.size() == result.alternate_jobs.size();
  }
  CaptureBufferOwners(state->prepared_buffers, result.prepared_buffers,
                      result.valid);
  result.resources.reserve(state->resources.size());
  for (const PipelineResource &resource : state->resources) {
    result.resources.push_back(resource.buffer.get());
    result.valid = result.valid && resource.buffer != nullptr;
  }
  result.claims.reserve(state->claims.size());
  for (const BufferClaim claim : state->claims) {
    result.claims.push_back(claim.buffer);
    result.valid = result.valid && claim.buffer != nullptr;
  }
  result.alternate_claims.reserve(state->alternate_claims.size());
  for (const BufferClaim claim : state->alternate_claims) {
    result.alternate_claims.push_back(claim.buffer);
    result.valid = result.valid && claim.buffer != nullptr;
  }
  result.valid = result.valid && state->publication != nullptr;
  const std::span<const PipelineStatePair> state_pairs =
      state->publication == nullptr
          ? std::span<const PipelineStatePair>{}
          : std::span<const PipelineStatePair>{state->publication->state_pairs};
  result.state_banks.reserve(state_pairs.size() * 2u);
  for (const PipelineStatePair &pair : state_pairs) {
    result.state_banks.push_back(pair.first.get());
    result.state_banks.push_back(pair.second.get());
    result.valid =
        result.valid && pair.first != nullptr && pair.second != nullptr;
  }
  result.publications.reserve(state->publications.size() * 2u);
  for (const PipelinePublicationPlan &publication : state->publications) {
    const PipelinePublicationViewPlan *source = nullptr;
    if (const auto *terminal =
            std::get_if<PipelineTerminalPublicationPlan>(&publication)) {
      const std::uint32_t final =
          terminal->state < state->windows.size()
              ? state->windows[terminal->state].control.final
              : std::numeric_limits<std::uint32_t>::max();
      source = final < terminal->sources.size() ? &terminal->sources[final]
                                                : nullptr;
    } else {
      source = &std::get<PipelineWindowPublicationPlan>(publication).source;
    }
    const PipelinePublicationTargetPlan &target =
        pipeline_publication_target(publication);
    const std::uint32_t source_ordinal =
        source == nullptr ? PipelineResource::no_output
                          : source->identity.resource_ordinal;
    const std::uint32_t target_ordinal = target.view.identity.resource_ordinal;
    const BufferState *const source_owner =
        source_ordinal < state->resources.size()
            ? state->resources[source_ordinal].buffer.get()
            : nullptr;
    const BufferState *const target_owner =
        target_ordinal < state->resources.size()
            ? state->resources[target_ordinal].buffer.get()
            : nullptr;
    result.publications.push_back(source_owner);
    result.publications.push_back(target_owner);
    result.valid =
        result.valid && source_owner != nullptr && target_owner != nullptr;
  }
  result.window_counts.reserve(state->windows.size());
  for (std::size_t index = 0; index < state->windows.size(); ++index) {
    const PipelineWindow &window = state->windows[index];
    const std::uint32_t ordinal =
        window.control.count.identity.resource_ordinal;
    const BufferState *const owner =
        ordinal < state->resources.size()
            ? state->resources[ordinal].buffer.get()
            : nullptr;
    result.window_counts.push_back(owner);
    result.valid = result.valid && owner != nullptr;
  }
  return result;
}

} // namespace rund::node::test_contract::window
