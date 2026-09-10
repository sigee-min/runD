#include "local.hpp"

#include <limits>
#include <mutex>

namespace rund_node_test_pipeline::checkpoint {

int CheckBoundaries(Context &context) {
  using namespace rund::compute;

  // A non-state Pipeline has no resident checkpoint authority.
  auto ordinary_input = Upload(context.device, Initial);
  auto ordinary_output = context.device.buffer<std::int32_t>(Initial.size());
  auto ordinary = ordinary_input && ordinary_output
                      ? pipeline(context.device)
                            .then(context.advance, read(*ordinary_input),
                                  write(*ordinary_output))
                            .prepare()
                      : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const auto no_latest =
      ordinary ? ordinary->latest_device_state()
               : Result<LatestDeviceState>::fail(Reason::PipelineInvalid);
  const auto no_storage =
      ordinary ? ordinary->snapshot_storage()
               : Result<SnapshotStorage>::fail(Reason::PipelineInvalid);
  if (!ordinary || no_latest || no_latest.reason() != Reason::PipelineInvalid ||
      no_storage || no_storage.reason() != Reason::PipelineInvalid) {
    return 33;
  }

  // The public 64-bit generation stops at the exact 32-bit native control
  // capacity before claims, writes, or submission. Oversized restore rejects
  // before inspecting untrusted snapshot layout.
  auto capacity_first = Upload(context.device, Initial);
  auto capacity_second = context.device.buffer<std::int32_t>(Initial.size());
  auto capacity = capacity_first && capacity_second
                      ? pipeline(context.device)
                            .state(*capacity_first, *capacity_second)
                            .then(context.advance, read(*capacity_first),
                                  write(*capacity_second))
                            .commit()
                            .prepare()
                      : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> capacity_state =
      capacity ? detail::PipelineStateAccess::state(*capacity)
               : std::shared_ptr<detail::PipelineState>{};
  if (!capacity || capacity_state == nullptr ||
      capacity_state->publication == nullptr) {
    return 34;
  }
  {
    std::lock_guard lock{capacity_state->publication->gate};
    capacity_state->publication->generation = PipelineGenerationCapacity;
  }
  const Status capacity_run = capacity->run();
  auto oversized = std::make_shared<detail::StateSnapshotState>();
  oversized->generation = std::numeric_limits<std::uint64_t>::max();
  const Status capacity_restore =
      detail::restore_pipeline_state(capacity_state, oversized);
  if (capacity_run || capacity_run.reason() != Reason::PipelineCapacity ||
      capacity_restore ||
      capacity_restore.reason() != Reason::PipelineCapacity ||
      capacity->generation() != PipelineGenerationCapacity ||
      capacity->poisoned() || capacity->stats().command_submits != 0u ||
      !ReadExact(*capacity, *capacity_first, context.observed) ||
      context.observed != Initial) {
    return 35;
  }

  // Observation identity must never become ambiguous through payload-epoch
  // wrap. Every payload-changing route fails before claims or submission at
  // UINT64_MAX, while the exact same-authority zero-byte rebase remains valid.
  auto epoch_first = Upload(context.device, Initial);
  auto epoch_second = context.device.buffer<std::int32_t>(Initial.size());
  auto epoch =
      epoch_first && epoch_second
          ? pipeline(context.device)
                .state(*epoch_first, *epoch_second)
                .then(context.advance, read(*epoch_first), write(*epoch_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> epoch_state =
      epoch ? detail::PipelineStateAccess::state(*epoch)
            : std::shared_ptr<detail::PipelineState>{};
  auto epoch_latest =
      epoch ? epoch->latest_device_state()
            : Result<LatestDeviceState>::fail(Reason::PipelineInvalid);
  if (!epoch || epoch_state == nullptr || epoch_state->publication == nullptr ||
      !epoch_latest) {
    return 36;
  }
  {
    std::lock_guard lock{epoch_state->publication->gate};
    epoch_state->publication->payload_epoch =
        std::numeric_limits<std::uint64_t>::max();
  }
  const Status epoch_noop = epoch->restore(*epoch_latest);
  const Status epoch_host = epoch->restore(context.storage);
  const Status epoch_copy = epoch->restore(context.latest);
  const Status epoch_run = epoch->run();
  if (!epoch_noop || epoch_host ||
      epoch_host.reason() != Reason::PipelineCapacity || epoch_copy ||
      epoch_copy.reason() != Reason::PipelineCapacity || epoch_run ||
      epoch_run.reason() != Reason::PipelineCapacity || epoch->poisoned() ||
      epoch->generation() != 0u || epoch->stats().command_submits != 0u ||
      epoch_state->publication->payload_epoch !=
          std::numeric_limits<std::uint64_t>::max() ||
      !ReadExact(*epoch, *epoch_first, context.observed) ||
      context.observed != Initial) {
    return 36;
  }

  // Exercise enough alternating commits to cross the native parity streams
  // repeatedly. The live selector and both reusable-bank identities must
  // follow every publication, not merely the first hand-off.
  constexpr std::uint64_t many_commits = 65u;
  const CheckpointStats many_before = context.prepared.checkpoint_stats();
  std::uint64_t previous_hash = context.storage.hash();
  for (std::uint64_t commit = 1u; commit <= many_commits; ++commit) {
    if (!context.prepared.run() ||
        !context.prepared.snapshot_into(context.storage) ||
        context.prepared.generation() != 3u + commit ||
        context.latest.generation() != 3u + commit ||
        context.storage.generation() != 3u + commit ||
        context.storage.hash() == 0u ||
        context.storage.hash() == previous_hash) {
      return 37;
    }
    previous_hash = context.storage.hash();
  }
  const CheckpointStats many_after = context.prepared.checkpoint_stats();
  std::array<std::int32_t, Initial.size()> many_published{};
  for (std::size_t index = 0u; index < Initial.size(); ++index) {
    many_published[index] =
        Initial[index] + 3 + static_cast<std::int32_t>(many_commits);
  }
  // Public transactional reads intentionally project either declared owner
  // onto the current publication selector; the inactive bank is not a
  // separately observable Pipeline result.
  if (many_after.reusable_snapshot_count -
              many_before.reusable_snapshot_count !=
          many_commits ||
      many_after.reusable_snapshot_byte_count -
              many_before.reusable_snapshot_byte_count !=
          many_commits * sizeof(Initial) ||
      many_after.reusable_snapshot_transfer_count -
              many_before.reusable_snapshot_transfer_count !=
          (context.backend == Backend::Cpu ? 0u : many_commits) ||
      !ReadExact(context.prepared, context.second, context.observed) ||
      context.observed != many_published ||
      !ReadExact(context.prepared, context.first, context.observed) ||
      context.observed != many_published) {
    return 37;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::checkpoint
