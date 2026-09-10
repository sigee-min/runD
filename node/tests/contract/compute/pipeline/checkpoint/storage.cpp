#include "local.hpp"

#include <utility>

namespace rund_node_test_pipeline::checkpoint {

int CheckStorageCapacity(Context &context) {
  using namespace rund::compute;

  // A previously valid bank survives both byte-capacity and field-capacity
  // failures byte-for-byte.
  constexpr std::array<std::int32_t, 2u> tiny_initial{9, 10};
  auto tiny_advance =
      on(context.device)
          .map<std::int32_t>("pipeline-reusable-checkpoint-tiny",
                             tiny_initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto tiny_first = Upload(context.device, tiny_initial);
  auto tiny_second = context.device.buffer<std::int32_t>(tiny_initial.size());
  auto tiny =
      tiny_advance && tiny_first && tiny_second
          ? pipeline(context.device)
                .state(*tiny_first, *tiny_second)
                .then(*tiny_advance, read(*tiny_first), write(*tiny_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  auto roomy_result =
      tiny ? tiny->snapshot_storage(64u)
           : Result<SnapshotStorage>::fail(Reason::PipelineInvalid);
  if (!tiny || !roomy_result) {
    return 24;
  }
  SnapshotStorage roomy = std::move(roomy_result).value();
  if (!tiny->snapshot_into(roomy) || !roomy.has_snapshot()) {
    return 25;
  }
  const std::uint64_t retained_generation = roomy.generation();
  const std::uint64_t retained_hash = roomy.hash();
  const graph::Fingerprint retained_fingerprint = roomy.fingerprint();
  const CheckpointStats before_capacity_failure =
      context.prepared.checkpoint_stats();

  auto pair_first = Upload(context.device, Initial);
  auto pair_second = context.device.buffer<std::int32_t>(Initial.size());
  auto pair_other_first = Upload(context.device, Initial);
  auto pair_other_second = context.device.buffer<std::int32_t>(Initial.size());
  auto paired =
      pair_first && pair_second && pair_other_first && pair_other_second
          ? pipeline(context.device)
                .state(*pair_first, *pair_second)
                .state(*pair_other_first, *pair_other_second)
                .then(context.advance, read(*pair_first), write(*pair_second))
                .then(context.advance, read(*pair_other_first),
                      write(*pair_other_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const Status byte_failure = context.prepared.snapshot_into(context.small);
  const Status field_failure = paired ? paired->snapshot_into(roomy)
                                      : Status::fail(Reason::PipelineInvalid);
  if (!paired || byte_failure ||
      byte_failure.reason() != Reason::BufferCapacity || field_failure ||
      field_failure.reason() != Reason::BufferCapacity ||
      !roomy.has_snapshot() || roomy.generation() != retained_generation ||
      roomy.hash() != retained_hash ||
      roomy.fingerprint() != retained_fingerprint ||
      !SameCheckpointStats(before_capacity_failure,
                           context.prepared.checkpoint_stats())) {
    return 26;
  }
  auto paired_storage_result = paired->snapshot_storage();
  if (!paired_storage_result) {
    return 27;
  }
  SnapshotStorage paired_storage = std::move(paired_storage_result).value();
  const auto paired_snapshot = paired->snapshot();
  if (!paired->snapshot_into(paired_storage) || !paired_snapshot ||
      paired_storage.field_capacity() != 2u ||
      paired_storage.capacity() != sizeof(Initial) * 2u ||
      paired_storage.hash() != paired_snapshot->hash()) {
    return 28;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::checkpoint
