#include "arena.hpp"
#include "local.hpp"

#include "../../buffer/local.hpp"
#include "../../job/local.hpp"
#include "../../memory/arena.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Result<std::shared_ptr<const PipelineMemoryPlan>>
plan_memory(const PipelineBuildState &build) {
  try {
    auto result = std::make_shared<PipelineMemoryPlan>();
    PipelinePlan &summary = result->summary;
    std::unordered_map<const BufferState *, std::uint8_t> external;
    external.reserve(std::min(build.binding_count, PipelineResourceCapacity));
    const auto admit = [&](const PipelineBinding &binding) {
      if (binding.owner != PipelineBinding::external) {
        return true;
      }
      if (binding.buffer == nullptr) {
        return false;
      }
      const auto [position, inserted] =
          external.emplace(binding.buffer.get(), 0u);
      return !inserted || kernel::checked::add(summary.persistent_bytes,
                                               binding.buffer->bytes,
                                               summary.persistent_bytes);
    };
    for (const PipelineBuildStep &step : build.steps) {
      for (const PipelineBinding &binding : step.inputs) {
        if (!admit(binding)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
      for (const PipelineBinding &binding : step.outputs) {
        if (!admit(binding)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
    }
    for (const PipelineBuildStatePair &pair : build.state_pairs) {
      if (!admit(pair.published) || !admit(pair.pending)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    for (const PipelineBuildPublish &publication : build.publications) {
      if (!admit(publication.source) || !admit(publication.target)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
      std::uint64_t bytes = 0u;
      if (publication.source.element_bytes == 0u ||
          !kernel::checked::mul(
              static_cast<std::uint64_t>(publication.source.count),
              publication.source.element_bytes, bytes) ||
          !kernel::checked::add(summary.publish_bytes, bytes,
                                summary.publish_bytes) ||
          !kernel::checked::add(summary.publish_count, 1u,
                                summary.publish_count)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }

    for (const PipelineInternal &internal : build.internals) {
      const std::size_t width = type_bytes(internal.type);
      std::uint64_t bytes = 0u;
      if (width == 0u ||
          !kernel::checked::mul(static_cast<std::uint64_t>(internal.count),
                                width, bytes) ||
          !kernel::checked::add(summary.state_bytes, bytes,
                                summary.state_bytes)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }

    const Status arena =
        plan_pipeline_arena(*build.device, build.steps, *result);
    if (!arena) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          arena.reason());
    }
    const Status views =
        plan_pipeline_views(*build.device, build.steps, *result);
    if (!views) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          views.reason());
    }
    const Status scratch =
        plan_pipeline_scratch(*build.device, build.steps, *result);
    if (!scratch) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          scratch.reason());
    }
    summary.allocation_count = build.internals.size() + result->chunks.size() +
                               result->view_chunks.size() +
                               result->scratch_chunks.size();
    summary.peak_bytes = summary.state_bytes;
    if (!kernel::checked::add(summary.peak_bytes, summary.transient_bytes,
                              summary.peak_bytes) ||
        !kernel::checked::add(summary.peak_bytes, summary.prepared_bytes,
                              summary.peak_bytes) ||
        !kernel::checked::add(summary.persistent_bytes, summary.peak_bytes,
                              summary.total_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::success(
        std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
        Reason::PipelineCapacity);
  }
}

[[nodiscard]] Status materialize_pipeline(PipelineBuildState &build) {
  try {
    std::vector<std::shared_ptr<BufferState>> owners;
    owners.reserve(build.internals.size());
    std::array<std::uint32_t, PipelineIterationCapacity> ordinals{};
    for (std::size_t index = 0u; index < build.internals.size(); ++index) {
      const PipelineInternal &internal = build.internals[index];
      auto made = make_input_binding_buffer(build.device, internal.type,
                                            internal.count);
      if (!made) {
        return Status::fail(made.reason());
      }
      if (internal.fill == PipelineFill::Ordinal) {
        if (internal.type != Type::U32 || internal.count > ordinals.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        std::iota(ordinals.begin(), ordinals.begin() + internal.count, 0u);
        WriteStats writes{};
        const Status written = write_buffer(*made,
                                            HostView{.data = ordinals.data(),
                                                     .count = internal.count,
                                                     .type = Type::U32},
                                            writes);
        if (!written) {
          return written;
        }
      }
      owners.push_back(std::move(made).value());
    }
    const auto resolve = [&](PipelineBinding &binding) {
      if (binding.owner == PipelineBinding::external) {
        return binding.buffer != nullptr;
      }
      if (binding.owner >= owners.size() || binding.buffer != nullptr) {
        return false;
      }
      binding.buffer = owners[binding.owner];
      return true;
    };
    for (PipelineBuildStep &step : build.steps) {
      for (PipelineBinding &binding : step.inputs) {
        if (!resolve(binding)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      for (PipelineBinding &binding : step.outputs) {
        if (!resolve(binding)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
    for (PipelineBuildPublish &publication : build.publications) {
      if (!resolve(publication.source) || !resolve(publication.target)) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

[[nodiscard]] Result<PipelineMemorySet>
make_pipeline_memory(const std::shared_ptr<DeviceState> &device,
                     const std::span<const PipelineBuildStep> steps,
                     const PipelineMemoryPlan &plan) {
  try {
    PipelineMemorySet result;
    result.steps.resize(steps.size());
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
      auto made = make_workspace_buffer(device, count);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.buffers.push_back(std::move(made).value());
    }
    if (plan.views.size() != steps.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.prepared.reserve(plan.view_chunks.size() +
                            plan.scratch_chunks.size());
    if (!plan.view_chunks.empty() || !plan.scratch_chunks.empty()) {
      result.arena = std::make_shared<JobArena>();
      result.arena->buffers.reserve(plan.view_chunks.size() +
                                    plan.scratch_chunks.size());
      result.arena->slots.reserve(plan.view_slots.size() +
                                  plan.scratch_chunks.size());
    }
    for (const std::size_t words : plan.view_chunks) {
      if (words == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
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
    for (const std::size_t words : plan.scratch_chunks) {
      if (words == 0u || result.arena == nullptr) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
      const std::size_t slot = result.arena->slots.size();
      result.arena->slots.push_back(
          JobArenaSlot{.words = words,
                       .owner = result.arena->buffers.size() - 1u,
                       .offset_words = 0u});
      std::uint64_t bytes = 0u;
      if (!kernel::checked::mul(static_cast<std::uint64_t>(words), memory::Word,
                                bytes)) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineCapacity);
      }
      result.arena->scratch.push_back(
          node::accel::detail::KernelScratchPage{.slot = slot, .bytes = bytes});
    }
    for (std::size_t step_index = 0u; step_index < steps.size(); ++step_index) {
      const PipelineBuildStep &step = steps[step_index];
      const auto &chunks = step.program->chunks;
      if (step.iteration_bound > 1u && step.iteration != 0u) {
        if (step_index == 0u || result.steps[step_index - 1u] == nullptr ||
            result.steps[step_index - 1u]->program != step.program) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        result.steps[step_index] = result.steps[step_index - 1u];
        continue;
      }
      if (chunks.empty() && plan.views[step_index].empty() &&
          step.iteration_bound == 1u && result.arena == nullptr) {
        continue;
      }
      const std::vector<std::uint32_t> &order = step.program->chunk_order;
      if (order.size() != chunks.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto workspace = std::make_shared<JobWorkspace>();
      workspace->program = step.program;
      workspace->arena = result.arena;
      workspace->buffers.resize(chunks.size());
      workspace->offsets.resize(chunks.size());
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
