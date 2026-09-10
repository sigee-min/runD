#include "local.hpp"

namespace rund_node_test_pipeline::checkpoint {

int CheckBusyAndCopy(Context &context) {
  using namespace rund::compute;
  Pipeline &prepared = context.prepared;
  SnapshotStorage &storage = context.storage;
  LatestDeviceState &latest = context.latest;

  // One reservation belongs to the shared publication authority, not to one
  // Pipeline wrapper. Busy operations preserve the active storage bank.
  const std::uint64_t busy_generation = storage.generation();
  const std::uint64_t busy_hash = storage.hash();
  if (!detail::queue_pipeline(context.source_state)) {
    return 17;
  }
  const Status busy_snapshot = prepared.snapshot_into(storage);
  const Status busy_peer = context.parity_one->run();
  const Status busy_restore = prepared.restore(storage);
  const Status cancelled = detail::cancel_pipeline(context.source_state);
  if (busy_snapshot || busy_snapshot.reason() != Reason::PipelineBusy ||
      busy_peer || busy_peer.reason() != Reason::PipelineBusy || busy_restore ||
      busy_restore.reason() != Reason::PipelineBusy || cancelled ||
      cancelled.reason() != Reason::Cancelled ||
      storage.generation() != busy_generation || storage.hash() != busy_hash ||
      latest.generation() != 3u || prepared.poisoned()) {
    return 18;
  }

  // Compatible disjoint owners take exactly one selected field, B bytes, to
  // destination parity zero without any host upload/download accounting.
  auto copy_first = context.device.buffer<std::int32_t>(Initial.size());
  auto copy_second = context.device.buffer<std::int32_t>(Initial.size());
  auto copied =
      copy_first && copy_second
          ? pipeline(context.device)
                .state(*copy_first, *copy_second)
                .then(context.advance, read(*copy_first), write(*copy_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!copied) {
    return 19;
  }
  const Stats before_copy = copied->stats();
  const Status copied_latest = copied->restore(latest);
  const Stats after_copy = copied->stats();
  const CheckpointStats copied_stats = copied->checkpoint_stats();
  const std::uint64_t expected_copy_commands =
      context.backend == Backend::Cpu ? 0u : 1u;
  if (!copied_latest || copied->generation() != 3u || copied->poisoned() ||
      copied_stats.device_state_copy_byte_count != sizeof(Initial) ||
      copied_stats.device_state_copy_command_count != expected_copy_commands ||
      after_copy.uploaded_bytes != before_copy.uploaded_bytes ||
      after_copy.downloaded_bytes != before_copy.downloaded_bytes ||
      after_copy.download_events != before_copy.download_events ||
      !ReadExact(*copied, *copy_first, context.observed) ||
      context.observed != Thrice || !copied->run() ||
      copied->generation() != 4u ||
      !ReadExact(*copied, *copy_second, context.observed) ||
      context.observed != Fourth) {
    return 20;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::checkpoint
