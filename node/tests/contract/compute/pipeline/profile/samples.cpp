#include "../local.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/sample.hpp"
#include "src/compute/pipeline/state.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>

namespace rund_node_test_pipeline {
namespace {

[[nodiscard]] constexpr bool SampleCleanAxesAreExhaustive() {
  using rund::compute::Stats;
  using rund::compute::detail::pipeline_sample_is_clean;
  constexpr std::array axes{
      &Stats::pipeline_compiles,
      &Stats::buffer_allocations,
      &Stats::buffer_reuses,
      &Stats::download_events,
      &Stats::uploaded_bytes,
      &Stats::downloaded_bytes,
      &Stats::pipeline_cache_hits,
      &Stats::pipeline_cache_evictions,
      &Stats::descriptor_pool_creations,
      &Stats::descriptor_set_allocations,
      &Stats::descriptor_reuses,
      &Stats::command_capacity_rejections,
      &Stats::shader_compile_ns,
      &Stats::spirv_compile_ns,
      &Stats::pipeline_create_ns,
      &Stats::descriptor_setup_ns,
      &Stats::readback_ns,
      &Stats::output_hash,
      &Stats::host_write_bytes,
  };
  if (!pipeline_sample_is_clean(Stats{})) {
    return false;
  }
  for (const auto axis : axes) {
    Stats dirty{};
    dirty.*axis = 1u;
    if (pipeline_sample_is_clean(dirty)) {
      return false;
    }
  }
  Stats transfer{};
  transfer.transfer_submissions.host_to_device = 1u;
  if (pipeline_sample_is_clean(transfer)) {
    return false;
  }
  transfer = {};
  transfer.transfer_submissions.device_to_host = 1u;
  if (pipeline_sample_is_clean(transfer)) {
    return false;
  }
  transfer = {};
  transfer.transfer_submissions.device_to_device = 1u;
  return !pipeline_sample_is_clean(transfer);
}

static_assert(SampleCleanAxesAreExhaustive());

std::atomic<bool> write_entered{};
std::atomic<bool> release_write{};

rund::compute::detail::UploadResult
BlockingUpload(rund::compute::detail::DeviceState &,
               rund::compute::detail::BufferState &, const void *,
               std::size_t) {
  write_entered.store(true, std::memory_order_release);
  while (!release_write.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  return rund::compute::detail::UploadResult{
      .status = rund::compute::Status::success(),
      .staging_bytes = 16u,
      .staging_peak_bytes = 16u,
      .staging_budget = 64u,
      .buffer_allocations = 1u,
      .command_submits = 1u,
  };
}

[[nodiscard]] bool SampleWriteIsSerialized() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  static const DeviceOps operations{.upload = BlockingUpload};
  auto device = std::make_shared<DeviceState>();
  device->backend = Backend::Vulkan;
  device->ops = &operations;
  auto state = std::make_shared<PipelineState>();
  state->device = device;
  state->steps.emplace_back();
  auto buffer = std::make_shared<BufferState>();
  buffer->device = device;
  buffer->type = Type::U32;
  buffer->count = 4u;
  buffer->bytes = 16u;
  constexpr std::array<std::uint32_t, 4u> values{1u, 2u, 3u, 4u};
  WriteStats writes{};
  Status written = Status::fail(Reason::PipelineInvalid);
  write_entered.store(false, std::memory_order_relaxed);
  release_write.store(false, std::memory_order_relaxed);
  if (!begin_pipeline_samples(state)) {
    return false;
  }
  std::thread writer{[&] {
    written = write_pipeline_raw(
        state, buffer, HostView{values.data(), values.size(), Type::U32},
        writes);
  }};
  while (!write_entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  const Status concurrent_end = end_pipeline_samples(state);
  release_write.store(true, std::memory_order_release);
  writer.join();
  const Status ended = end_pipeline_samples(state);
  return concurrent_end.reason() == Reason::ProfileBusy && written && ended &&
         state->samples == PipelineState::SampleState::Inactive &&
         writes.uploads == 1u && writes.bytes == buffer->bytes &&
         state->stats.host_write_bytes == buffer->bytes &&
         state->stats.uploaded_bytes == buffer->bytes &&
         state->stats.transfer_submissions.host_to_device == 1u &&
         state->stats.buffer_allocations == 1u && state->staging_bytes == 16u &&
         state->staging_peak == 16u && state->staging_budget == 64u;
}

[[nodiscard]] bool
SampleCheckpointBoundariesAreExclusive(rund::compute::Device &device,
                                       const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> initial{1, 2, 3, 4};
  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-sample-checkpoint", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = Upload(device, initial);
  auto second = device.buffer<std::int32_t>(initial.size());
  auto prepared = advance && first && second
                      ? pipeline(device)
                            .state(*first, *second)
                            .then(*advance, read(*first), write(*second))
                            .commit()
                            .prepare()
                      : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!prepared) {
    return false;
  }
  auto storage_result = prepared->snapshot_storage();
  auto saved = prepared->snapshot();
  auto latest = prepared->latest_device_state();
  if (!storage_result || !saved || !latest) {
    return false;
  }
  SnapshotStorage storage = std::move(storage_result).value();
  if (!prepared->snapshot_into(storage) || !storage.has_snapshot() ||
      !prepared->begin_samples() || !prepared->run()) {
    return false;
  }

  const auto blocked_snapshot = prepared->snapshot();
  const auto blocked_storage = prepared->snapshot_storage();
  const auto blocked_latest = prepared->latest_device_state();
  const Status blocked_into = prepared->snapshot_into(storage);
  const Status blocked_saved_restore = prepared->restore(*saved);
  const Status blocked_storage_restore = prepared->restore(storage);
  const Status blocked_latest_restore = prepared->restore(*latest);
  if (blocked_snapshot.reason() != Reason::ProfileBusy ||
      blocked_storage.reason() != Reason::ProfileBusy ||
      blocked_latest.reason() != Reason::ProfileBusy ||
      blocked_into.reason() != Reason::ProfileBusy ||
      blocked_saved_restore.reason() != Reason::ProfileBusy ||
      blocked_storage_restore.reason() != Reason::ProfileBusy ||
      blocked_latest_restore.reason() != Reason::ProfileBusy ||
      !prepared->run() || !prepared->end_samples() ||
      prepared->generation() != 2u) {
    return false;
  }
  const auto profile = prepared->profile();
  return profile && profile->execution().backend == backend &&
         profile->execution().pipeline.samples_clean(2u);
}

} // namespace

[[nodiscard]] int CheckProfileSamples(rund::compute::Device &device,
                                      const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> input{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 4u> expected{2u, 3u, 4u, 5u};
  auto program =
      on(device)
          .map<std::uint32_t>("pipeline-profile-samples", input.size(),
                              [](auto value) { return value + 1u; })
          .compile();
  auto source = Upload(device, input);
  auto output = device.buffer<std::uint32_t>(input.size());
  auto prepared = program && source && output
                      ? pipeline(device)
                            .then(*program, read(*source), write(*output))
                            .prepare()
                      : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!prepared || prepared->begin_samples().reason() != Reason::Ok) {
    return 1;
  }
  if (prepared->begin_samples().reason() != Reason::ProfileBusy) {
    return 2;
  }
  if (prepared->profile().reason() != Reason::ProfileBusy) {
    return 3;
  }
  for (std::uint32_t sample = 0u; sample != 3u; ++sample) {
    if (!prepared->run()) {
      return 4;
    }
  }
  if (!prepared->end_samples()) {
    return 5;
  }
  std::array<std::uint32_t, 4u> observed{};
  if (!ReadExact(*prepared, *output, observed) || observed != expected) {
    return 6;
  }
  const auto clean = prepared->profile();
  if (!clean || clean->execution().backend != backend ||
      !clean->execution().pipeline.samples_clean(3u) ||
      clean->execution().pipeline.sampled_runs != 3u ||
      clean->execution().pipeline.clean_runs != 3u) {
    return 7;
  }

  if (!prepared->begin_samples() || !prepared->run()) {
    return 8;
  }
  observed.fill(0u);
  if (!ReadExact(*prepared, *output, observed) || observed != expected ||
      !prepared->run() || !prepared->end_samples()) {
    return 9;
  }
  const auto dirty = prepared->profile();
  if (!dirty || dirty->execution().pipeline.samples_clean(2u) ||
      dirty->execution().pipeline.sampled_runs != 2u ||
      dirty->execution().pipeline.clean_runs != 1u ||
      prepared->end_samples().reason() != Reason::ProfileInvalid) {
    return 10;
  }

  if (!prepared->begin_samples()) {
    return 11;
  }
  bool wrote = false;
  const Status feedback = host_feedback(
      *prepared, 2u, [&](HostIteration &iteration) noexcept -> Status {
        if (iteration.completed() == 1u) {
          wrote = true;
          return iteration.write(*source, std::span{input});
        }
        return Status::success();
      });
  if (!feedback || !wrote || !prepared->end_samples()) {
    return 12;
  }
  const auto written = prepared->profile();
  if (!written || written->execution().pipeline.samples_clean(2u) ||
      written->execution().pipeline.sampled_runs != 2u ||
      written->execution().pipeline.clean_runs != 1u) {
    return 13;
  }

  constexpr std::uint32_t saturated = std::numeric_limits<std::uint32_t>::max();
  constexpr PipelineStats capped{.sampled_runs = saturated,
                                 .clean_runs = saturated};
  static_assert(!capped.samples_clean(saturated));
  if (!SampleWriteIsSerialized()) {
    return 14;
  }
  if (!SampleCheckpointBoundariesAreExclusive(device, backend)) {
    return 15;
  }
  Pipeline moved = std::move(*prepared);
  if (prepared->begin_samples().reason() != Reason::ProfileInvalid ||
      prepared->end_samples().reason() != Reason::ProfileInvalid || !moved) {
    return 16;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
