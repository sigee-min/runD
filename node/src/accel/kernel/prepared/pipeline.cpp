#include "../prepared.hpp"

#include "evidence.hpp"
#include "model.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <span>

namespace rund::node::accel::detail {

PreparedKernelPipeline
PrepareKernelPipeline(const rund::AccelContext &context,
                      const std::span<const PreparedKernelRun *const> runs,
                      const std::span<const std::uint8_t> barriers,
                      const std::span<const std::uint32_t> declared_steps,
                      const std::span<const BackendRecurrence> recurrences,
                      const std::span<const BackendPublish> publications,
                      const std::uint32_t declared_step_count,
                      const std::uint32_t generation_stride,
                      const bool profile_steps) {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  if (runs.empty() || runs.size() > PreparedPipelineStepCapacity ||
      runs.size() != barriers.size() || runs.size() != declared_steps.size() ||
      runs.size() != recurrences.size() || publications.size() > 32u) {
    return {};
  }
  std::uint32_t state_count = 0u;
  for (const BackendRecurrence &recurrence : recurrences) {
    if (recurrence.window != nullptr) {
      state_count = std::max(state_count, recurrence.window->state + 1u);
    }
  }
  for (const BackendPublish &publication : publications) {
    const auto &target = publication.target;
    if (publication.target_handle == nullptr ||
        publication.state >= state_count ||
        publication.final >= publication.sources.size() ||
        target.stride_bytes < target.element_bytes ||
        target.usage != rund::kernel::kResidentUsageWrite) {
      return PreparedKernelPipeline{.reason = invalid.reason};
    }
    for (const BackendRead &read : publication.sources) {
      const auto &source = read.source;
      if (read.handle == nullptr || source.count != target.count ||
          source.element_bytes != target.element_bytes ||
          (source.element_bytes != 4u && source.element_bytes != 8u) ||
          source.stride_bytes < source.element_bytes ||
          source.usage != rund::kernel::kResidentUsageRead) {
        return PreparedKernelPipeline{.reason = invalid.reason};
      }
    }
  }
  std::shared_ptr<prepared::PipelineState> pipeline{};
  try {
    pipeline = std::make_shared<prepared::PipelineState>();
    pipeline->states =
        std::make_unique<std::shared_ptr<prepared::RunState>[]>(runs.size());
  } catch (const std::bad_alloc &) {
    return {};
  }
  pipeline->context = context;
  pipeline->size = runs.size();
  if (!PreparePipelineStatusLayout(pipeline->status, declared_steps,
                                   declared_step_count, generation_stride)) {
    return PreparedKernelPipeline{.reason = invalid.reason};
  }
  std::array<BackendBatchEntry, PreparedPipelineStepCapacity> entries{};
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    auto *const state =
        item == nullptr ? nullptr
                        : static_cast<prepared::RunState *>(item->owner.get());
    const BackendOps *const candidate =
        state == nullptr ? nullptr : state->bound.run.ops;
    if (item == nullptr || !item->ok || state == nullptr ||
        !IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
        candidate->prepare_pipeline == nullptr ||
        candidate->submit_prepared_pipeline == nullptr ||
        !prepared::MatchesContext(context, *state) ||
        (pipeline->ops != nullptr && pipeline->ops != candidate)) {
      return PreparedKernelPipeline{.reason = invalid.reason};
    }
    pipeline->ops = candidate;
    pipeline->states[index] =
        std::static_pointer_cast<prepared::RunState>(item->owner);
    prepared::Accumulate(pipeline->counts, *state);
    entries[index] = BackendBatchEntry{.run = &state->bound.run,
                                       .prepared = &state->backend,
                                       .recurrence = recurrences[index]};
  }
  const rund::AccelCheck built = pipeline->ops->prepare_pipeline(
      std::span<const BackendBatchEntry>{entries.data(), runs.size()}, barriers,
      publications, pipeline->status, profile_steps, pipeline->backend,
      pipeline->memory);
  if (!built.ok) {
    return PreparedKernelPipeline{.reason = built.reason};
  }
  if (!ValidPreparedPipelineStatusLayout(pipeline->status, declared_steps,
                                         declared_step_count,
                                         generation_stride)) {
    return PreparedKernelPipeline{.reason = invalid.reason};
  }
  const std::uint64_t common_host_bytes =
      sizeof(prepared::PipelineState) +
      pipeline->size * sizeof(std::shared_ptr<prepared::RunState>);
  accumulate_memory(pipeline->memory.host,
                    PreparedMemory{.current = common_host_bytes,
                                   .peak = common_host_bytes,
                                   .cumulative = common_host_bytes,
                                   .budget = common_host_bytes});
  return PreparedKernelPipeline{
      .owner = std::static_pointer_cast<void>(pipeline),
      .ok = true,
      .reason = "ok",
  };
}

rund::AccelCheck
SeedPreparedKernelPipelineGeneration(const PreparedKernelPipeline &prepared,
                                     const std::uint32_t generation) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->seed_prepared_pipeline_generation == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  return pipeline->ops->seed_prepared_pipeline_generation(pipeline->backend,
                                                          generation);
}

PreparedPipelineMemory ReadPreparedKernelPipelineMemory(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->memory
                                            : PreparedPipelineMemory{};
}

PreparedPipelineStatusLayout ReadPreparedKernelPipelineStatus(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->status
                                            : PreparedPipelineStatusLayout{};
}

} // namespace rund::node::accel::detail
