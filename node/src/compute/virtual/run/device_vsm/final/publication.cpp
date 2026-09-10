#include "../internal.hpp"

#include <algorithm>
#include <functional>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

[[nodiscard]] bool lock_pipelines(DeviceVsmProductRun &run,
                                  DeviceVsmPublication &publication) noexcept {
  if (run.owner == nullptr || run.owner->pipeline_count < 2u ||
      run.owner->pipeline_count > DeviceVsmPipelineCapacity ||
      run.snapshot.count != run.owner->pipeline_count) {
    return false;
  }
  std::array<PipelineState *, DeviceVsmPipelineCapacity> ordered{};
  for (std::size_t index = 0u; index < run.owner->pipeline_count; ++index) {
    ordered[index] = run.owner->pipelines[index].get();
    if (ordered[index] == nullptr ||
        std::find(ordered.begin(), ordered.begin() + index, ordered[index]) !=
            ordered.begin() + index) {
      return false;
    }
  }
  std::sort(ordered.begin(), ordered.begin() + run.owner->pipeline_count,
            std::less<PipelineState *>{});
  for (std::size_t index = 0u; index < run.owner->pipeline_count; ++index) {
    publication.pipeline_locks[index] =
        std::unique_lock<std::mutex>{ordered[index]->gate};
  }
  publication.pipeline_lock_count = run.owner->pipeline_count;
  publication.run = &run;
  for (std::size_t bank = 0u; bank < run.owner->pipeline_count; ++bank) {
    PipelineState &pipeline = *run.owner->pipelines[bank];
    if (!run.pipeline_started[bank] || pipeline.publication == nullptr ||
        pipeline.phase != PipelinePhase::Running || pipeline.control_poisoned ||
        pipeline.transactional ||
        pipeline.attempt.generation != run.snapshot.generation[bank] ||
        pipeline.attempt.parity != run.snapshot.parity[bank]) {
      return false;
    }
    std::lock_guard publication_lock{pipeline.publication->gate};
    if (pipeline.publication->device_lost ||
        !pipeline.publication->attempt_active ||
        pipeline.publication->generation != pipeline.attempt.generation ||
        pipeline.publication->parity != pipeline.attempt.parity) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool rebase_pipelines(DeviceVsmProductRun &run) noexcept {
  std::size_t seeded = 0u;
  for (; seeded < run.owner->pipeline_count; ++seeded) {
    const std::uint64_t generation = run.snapshot.generation[seeded];
    if (generation == std::numeric_limits<std::uint64_t>::max() ||
        !seed_pipeline_generations(*run.owner->pipelines[seeded],
                                   generation + 1u,
                                   run.snapshot.parity[seeded])) {
      break;
    }
  }
  if (seeded == run.owner->pipeline_count) {
    return true;
  }
  bool restored = true;
  while (seeded != 0u) {
    --seeded;
    restored = seed_pipeline_generations(*run.owner->pipelines[seeded],
                                         run.snapshot.generation[seeded],
                                         run.snapshot.parity[seeded]) &&
               restored;
  }
  if (!restored) {
    for (std::size_t index = 0u; index < run.owner->pipeline_count; ++index) {
      run.owner->pipelines[index]->control_poisoned = true;
    }
  }
  return false;
}

void publish_backing(void *const raw) noexcept {
  auto &publication = *static_cast<DeviceVsmPublication *>(raw);
  DeviceVsmProductRun &run = *publication.run;
  const bool success = static_cast<bool>(publication.status);
  if (success) {
    VirtualBackingAccess::publish_write(*run.output);
    VirtualBackingAccess::clear_recovery(*run.output);
    ++run.owner->evidence->backing_publication_count;
  } else if (!run.backing_may_write) {
    VirtualBackingAccess::clear_recovery(*run.output);
  }
  run.backing_recovery = !success && run.backing_may_write;
}

} // namespace

bool prepare_success_publication(DeviceVsmProductRun &run,
                                 DeviceVsmPublication &publication) noexcept {
  if (run.output == nullptr || run.projection == nullptr ||
      !run.backing_recovery ||
      VirtualBackingAccess::version(*run.output) !=
          run.projection->output_version ||
      VirtualBackingAccess::recovery_bytes(*run.output) !=
          run.projection->active.output_bytes ||
      !lock_pipelines(run, publication) || !rebase_pipelines(run)) {
    return false;
  }
  publication.status = Status::success();
  return true;
}

bool prepare_failure_publication(DeviceVsmProductRun &run,
                                 DeviceVsmPublication &publication,
                                 const Status failure) noexcept {
  if (failure || !lock_pipelines(run, publication)) {
    return false;
  }
  publication.status = failure;
  return true;
}

void commit_publication(void *const raw,
                        const bool authority_success) noexcept {
  auto &publication = *static_cast<DeviceVsmPublication *>(raw);
  DeviceVsmProductRun &run = *publication.run;
  const bool expected = static_cast<bool>(publication.status);
  if (authority_success != expected) {
    publication.status = Status::fail(Reason::CompletionInvalid);
  }
  const bool success = static_cast<bool>(publication.status);
  const std::size_t issued =
      success ? (run.owner->proof->topology ==
                         node::accel::detail::DeviceVsmTopology::GraphMapReduce
                     ? 1u
                     : static_cast<std::size_t>(run.projection->frame_capacity))
              : 0u;
  const PipelineTerminal terminal{
      .reason = publication.status.reason(),
      .verified = issued,
      .issued_steps = issued,
      .writes_possible = false,
      .publication_suppressed = !success,
  };
  if (run.projection->poolless_device_vsm()) {
    publish_shared_pipeline_terminal_pair(*run.owner->pipelines[0u], terminal,
                                          *run.owner->pipelines[1u], terminal,
                                          &publication, publish_backing);
  } else {
    std::array<PipelineState *, DeviceVsmPipelineCapacity> pipelines{};
    std::array<PipelineTerminal, DeviceVsmPipelineCapacity> terminals{};
    for (std::size_t index = 0u; index < run.owner->pipeline_count; ++index) {
      pipelines[index] = run.owner->pipelines[index].get();
      terminals[index] = terminal;
    }
    publish_private_pipeline_terminal_set(
        {pipelines.data(), run.owner->pipeline_count},
        {terminals.data(), run.owner->pipeline_count}, &publication,
        publish_backing);
  }
  run.pipeline_started = {};
  run.publication_success = success;
  ++run.owner->evidence->authority_accept_count;
  run.owner->evidence->pipeline_terminal_count += run.owner->pipeline_count;
}

} // namespace rund::compute::detail::device_vsm_product_detail
