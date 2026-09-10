#include "../claim.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <mutex>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status validate_deferred_pipeline_generation(
    const PipelineState &pipeline,
    const DeferredPipelinePublication commit) noexcept {
  if (pipeline.phase != PipelinePhase::Ready ||
      pipeline.publication == nullptr || pipeline.publication->attempt_active ||
      pipeline.publication->device_lost ||
      commit.base_generation >= PipelineGenerationCapacity ||
      commit.base_payload_epoch == std::numeric_limits<std::uint64_t>::max() ||
      commit.base_parity > 1u ||
      pipeline.publication->generation != commit.base_generation ||
      pipeline.publication->payload_epoch != commit.base_payload_epoch ||
      pipeline.publication->parity != commit.base_parity ||
      pipeline.stats.publication.generation != commit.base_generation) {
    return Status::fail(Reason::CompletionInvalid);
  }
  if (commit.terminal_count >
          std::numeric_limits<std::uint64_t>::max() - commit.base_generation ||
      commit.terminal_count > std::numeric_limits<std::uint64_t>::max() -
                                  commit.base_payload_epoch) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const std::uint64_t generation =
      commit.base_generation + commit.terminal_count;
  if (generation > PipelineGenerationCapacity ||
      commit.control_generation != generation ||
      pipeline.native_generation != commit.control_generation ||
      pipeline.native_parity != commit.base_parity) {
    return Status::fail(Reason::CompletionInvalid);
  }
  if (commit.terminal_count != 0u &&
      (pipeline.attempt.generation ==
           std::numeric_limits<std::uint64_t>::max() ||
       pipeline.attempt.generation + 1u != commit.control_generation ||
       pipeline.attempt.parity != commit.base_parity)) {
    return Status::fail(Reason::CompletionInvalid);
  }
  return Status::success();
}

[[nodiscard]] Status
validate_deferred_pipeline_pair(PipelineState &first,
                                PipelineState &second) noexcept {
  if (&first == &second || first.device == nullptr ||
      first.device != second.device || first.publication == nullptr ||
      second.publication == nullptr ||
      first.publication == second.publication || first.transactional ||
      second.transactional || !has_private_residency_authority(first) ||
      !has_private_residency_authority(second)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

void lock_deferred_pipeline_pair(
    PipelineState &first, PipelineState &second,
    std::array<std::unique_lock<std::mutex>, 2u> &pipeline_locks,
    std::array<std::unique_lock<std::mutex>, 2u> &publication_locks) noexcept {
  std::array<PipelineState *, 2u> ordered_pipelines{&first, &second};
  std::sort(ordered_pipelines.begin(), ordered_pipelines.end(),
            std::less<PipelineState *>{});
  for (std::size_t index = 0u; index < ordered_pipelines.size(); ++index) {
    pipeline_locks[index] =
        std::unique_lock<std::mutex>{ordered_pipelines[index]->gate};
  }
  std::array<PipelinePublicationState *, 2u> ordered_publications{
      first.publication.get(), second.publication.get()};
  std::sort(ordered_publications.begin(), ordered_publications.end(),
            std::less<PipelinePublicationState *>{});
  for (std::size_t index = 0u; index < ordered_publications.size(); ++index) {
    publication_locks[index] =
        std::unique_lock<std::mutex>{ordered_publications[index]->gate};
  }
}

} // namespace

Status preflight_deferred_pipeline_generations(
    PipelineState &first, const DeferredPipelinePublication first_commit,
    PipelineState &second,
    const DeferredPipelinePublication second_commit) noexcept {
  const Status shape = validate_deferred_pipeline_pair(first, second);
  if (!shape) {
    return shape;
  }
  std::array<std::unique_lock<std::mutex>, 2u> pipeline_locks{};
  std::array<std::unique_lock<std::mutex>, 2u> publication_locks{};
  lock_deferred_pipeline_pair(first, second, pipeline_locks, publication_locks);
  const Status first_valid =
      validate_deferred_pipeline_generation(first, first_commit);
  return first_valid
             ? validate_deferred_pipeline_generation(second, second_commit)
             : first_valid;
}

void apply_deferred_pipeline_generations(
    PipelineState &first, const DeferredPipelinePublication first_commit,
    PipelineState &second,
    const DeferredPipelinePublication second_commit) noexcept {
  std::array<std::unique_lock<std::mutex>, 2u> pipeline_locks{};
  std::array<std::unique_lock<std::mutex>, 2u> publication_locks{};
  lock_deferred_pipeline_pair(first, second, pipeline_locks, publication_locks);

  const auto apply = [](PipelineState &pipeline,
                        const DeferredPipelinePublication commit) noexcept {
    const std::uint64_t generation =
        commit.base_generation + commit.terminal_count;
    const std::uint64_t payload_epoch =
        commit.base_payload_epoch + commit.terminal_count;
    pipeline.publication->generation = generation;
    pipeline.publication->payload_epoch = payload_epoch;
    pipeline.native_generation = generation;
    pipeline.native_parity = commit.base_parity;
    pipeline.observation_generation = generation;
    pipeline.observation_payload_epoch = payload_epoch;
    pipeline.observation_parity = commit.base_parity;
    pipeline.observation_identity_valid = true;
    pipeline.stats.publication.generation = generation;
    pipeline.unobserved_outputs = pipeline.outputs.size();
    pipeline.stats.output_hash = 0u;
  };
  apply(first, first_commit);
  apply(second, second_commit);
}

} // namespace rund::compute::detail
