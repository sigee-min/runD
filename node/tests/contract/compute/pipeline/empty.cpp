#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/job/local.hpp"
#include "src/compute/pipeline/state.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckZeroWork(rund::compute::Device &device,
                                const Backend backend) {
  using namespace rund::compute;
  auto program = on(device)
                     .map<std::int32_t>("pipeline-empty-work", 0u,
                                        [](auto value) { return value + 1; })
                     .compile();
  const std::array<std::int32_t, 0u> empty{};
  auto input = Upload(device, empty);
  auto output = device.buffer<std::int32_t>(0u);
  if (!program || !input || !output) {
    return 1;
  }
  auto prepared =
      pipeline(device).then(*program, read(*input), write(*output)).prepare();
  const auto prepared_state =
      prepared ? detail::PipelineStateAccess::state(*prepared)
               : std::shared_ptr<detail::PipelineState>{};
  if (!prepared || prepared_state == nullptr ||
      prepared_state->steps.size() != 1u ||
      prepared_state->steps.front().job == nullptr ||
      prepared_state->steps.front().job->workspace != nullptr ||
      (backend == Backend::Cpu && !prepared_state->cpu_storage.empty()) ||
      !prepared->run()) {
    return 2;
  }
  std::array<std::int32_t, 0u> observed{};
  const Stats stats = prepared->stats();
  if (!ReadExact(*prepared, *output, observed) || stats.dispatches != 0u ||
      stats.command_submits != 0u || stats.pipeline.verified_step_count != 1u ||
      stats.backend != backend) {
    return 3;
  }

  auto repeated = pipeline(device)
                      .repeat<3u>(*program, read(*input), write_final(*output))
                      .prepare();
  const auto repeated_state =
      repeated ? detail::PipelineStateAccess::state(*repeated)
               : std::shared_ptr<detail::PipelineState>{};
  if (!repeated || repeated_state == nullptr ||
      repeated_state->steps.size() != 3u ||
      (backend == Backend::Cpu && !repeated_state->cpu_storage.empty())) {
    return 8;
  }
  for (const detail::PipelineStep &step : repeated_state->steps) {
    if (step.job == nullptr || step.job->workspace != nullptr) {
      return 8;
    }
  }
  if (!repeated->run()) {
    return 8;
  }

  // A Pipeline with no state declaration has no checkpoint authority, but a
  // declared zero-byte state pair is a distinct, valid metadata-only state.
  // Its generation and content identity must remain observable even though no
  // payload transfer or backend command is required.
  auto state_input = Upload(device, empty);
  auto state_output = device.buffer<std::int32_t>(0u);
  auto transactional =
      state_input && state_output
          ? pipeline(device)
                .state(*state_input, *state_output)
                .then(*program, read(*state_input), write(*state_output))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  auto storage_result =
      transactional ? transactional->snapshot_storage()
                    : Result<SnapshotStorage>::fail(Reason::PipelineInvalid);
  auto latest = transactional
                    ? transactional->latest_device_state()
                    : Result<LatestDeviceState>::fail(Reason::PipelineInvalid);
  if (!transactional || !storage_result || !latest) {
    return 4;
  }
  SnapshotStorage storage = std::move(storage_result).value();
  if (!storage.valid() || storage.has_snapshot() || storage.capacity() != 0u ||
      storage.field_capacity() != 1u || latest->generation() != 0u ||
      !transactional->snapshot_into(storage) || !storage.has_snapshot() ||
      storage.generation() != 0u || storage.hash() == 0u) {
    return 5;
  }
  const std::uint64_t initial_hash = storage.hash();
  const auto immutable = transactional->snapshot();
  if (!immutable || immutable->generation() != 0u ||
      immutable->hash() != initial_hash || !transactional->run() ||
      transactional->generation() != 1u || latest->generation() != 1u ||
      !transactional->snapshot_into(storage) || storage.generation() != 1u ||
      storage.hash() == initial_hash ||
      !ReadExact(*transactional, *state_output, observed)) {
    return 6;
  }
  const CheckpointStats checkpoint = transactional->checkpoint_stats();
  return checkpoint.reusable_snapshot_count == 2u &&
                 checkpoint.reusable_snapshot_byte_count == 0u &&
                 checkpoint.reusable_snapshot_transfer_count == 0u
             ? 0
             : 7;
}

} // namespace rund_node_test_pipeline
