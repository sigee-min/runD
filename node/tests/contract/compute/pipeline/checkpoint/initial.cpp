#include "local.hpp"

#include <utility>

namespace rund_node_test_pipeline::checkpoint {

int CheckInitialAndParity(Context &context) {
  using namespace rund::compute;
  Pipeline &prepared = context.prepared;
  SnapshotStorage &storage = context.storage;
  SnapshotStorage &small = context.small;
  LatestDeviceState &latest = context.latest;
  const auto &source_state = context.source_state;

  if (!storage.valid() || storage.has_snapshot() ||
      storage.capacity() != sizeof(Initial) || storage.field_capacity() != 1u ||
      storage.generation() != 0u || storage.hash() != 0u ||
      storage.fingerprint() || !small.valid() || small.has_snapshot() ||
      !latest.valid() || latest.generation() != 0u ||
      latest.fingerprint() != prepared.fingerprint() ||
      prepared.checkpoint_stats().device_state_acquire_count != 1u) {
    return 5;
  }
  const Status empty_restore = prepared.restore(storage);
  const Status too_small = prepared.snapshot_into(small);
  if (empty_restore || empty_restore.reason() != Reason::PipelineInvalid ||
      prepared.poisoned() || prepared.generation() != 0u || too_small ||
      too_small.reason() != Reason::BufferCapacity || small.has_snapshot()) {
    return 6;
  }
  if (!prepared.snapshot_into(storage) || !storage.has_snapshot() ||
      storage.generation() != 0u || storage.hash() == 0u ||
      storage.fingerprint() != prepared.fingerprint()) {
    return 7;
  }
  const auto initial_snapshot = prepared.snapshot();
  if (!initial_snapshot || initial_snapshot->generation() != 0u ||
      initial_snapshot->hash() != storage.hash() ||
      initial_snapshot->fingerprint() != storage.fingerprint()) {
    return 8;
  }

  // A cold parity-zero handoff adopts the one publication authority without
  // touching payload bytes.
  auto parity_zero =
      pipeline(context.device)
          .state(context.first, context.second)
          .then(context.advance, read(context.first), write(context.second))
          .restore(latest)
          .commit()
          .prepare();
  const std::shared_ptr<detail::PipelineState> parity_zero_state =
      parity_zero ? detail::PipelineStateAccess::state(*parity_zero)
                  : std::shared_ptr<detail::PipelineState>{};
  if (!parity_zero || parity_zero_state == nullptr ||
      parity_zero_state->publication != source_state->publication ||
      parity_zero->generation() != 0u ||
      parity_zero->checkpoint_stats().device_state_rebase_count != 1u ||
      parity_zero->checkpoint_stats().device_state_copy_byte_count != 0u) {
    return 9;
  }
  context.parity_zero_state = parity_zero_state;
  context.parity_zero.emplace(std::move(*parity_zero));
  if (!prepared.run() || prepared.generation() != 1u ||
      latest.generation() != 1u) {
    return 10;
  }

  // Parity one must seed the alternate native generation stream as the next
  // selected stream. This catches a generation-stride-two handoff bug.
  auto parity_one =
      pipeline(context.device)
          .state(context.first, context.second)
          .then(context.advance, read(context.first), write(context.second))
          .restore(latest)
          .commit()
          .prepare();
  const std::shared_ptr<detail::PipelineState> parity_one_state =
      parity_one ? detail::PipelineStateAccess::state(*parity_one)
                 : std::shared_ptr<detail::PipelineState>{};
  if (!parity_one || parity_one_state == nullptr ||
      parity_one_state->publication != source_state->publication ||
      parity_one->generation() != 1u || !parity_one->run() ||
      latest.generation() != 2u || prepared.generation() != 2u ||
      !context.parity_zero->run() || latest.generation() != 3u) {
    return 11;
  }
  context.parity_one.emplace(std::move(*parity_one));
  if (!ReadExact(prepared, context.second, context.observed) ||
      context.observed != Thrice) {
    return 12;
  }

  if (!prepared.snapshot_into(storage) || storage.generation() != 3u) {
    return 13;
  }
  const auto current_snapshot = prepared.snapshot();
  if (!current_snapshot || current_snapshot->generation() != 3u ||
      current_snapshot->hash() != storage.hash() ||
      current_snapshot->fingerprint() != storage.fingerprint()) {
    return 14;
  }

  // Resetting a shared publication from odd parity to parity zero at the same
  // generation makes every sibling wrapper's native selector stale. The next
  // run must compare both generation and parity before choosing its prepared
  // stream; generation alone would advance the stride-two stream to g + 2.
  auto parity_probe_result = prepared.snapshot_storage();
  if (!parity_probe_result) {
    return 14;
  }
  SnapshotStorage parity_probe = std::move(parity_probe_result).value();
  if (!prepared.restore(storage) || source_state->publication->parity != 0u ||
      context.parity_zero_state->native_generation != 3u ||
      context.parity_zero_state->native_parity != 1u ||
      !context.parity_zero->run() || context.parity_zero->generation() != 4u ||
      !context.parity_zero->snapshot_into(parity_probe) ||
      parity_probe.generation() != 4u ||
      parity_probe.hash() == storage.hash() || !prepared.restore(storage) ||
      prepared.generation() != 3u || source_state->publication->parity != 0u ||
      !ReadExact(*context.parity_zero, context.first, context.observed) ||
      context.observed != Thrice ||
      context.parity_zero->stats().output_hash != 0u ||
      !prepared.restore(parity_probe) || prepared.generation() != 4u ||
      !ReadExact(*context.parity_zero, context.first, context.observed) ||
      context.observed != Fourth ||
      context.parity_zero->stats().output_hash != 0u ||
      !prepared.restore(storage) || prepared.generation() != 3u) {
    return 14;
  }
  node_compute_allocation::Start();
  const Status reused = prepared.snapshot_into(storage);
  node_compute_allocation::Stop();
  const std::uint64_t snapshot_process_allocations =
      node_compute_allocation::Count();
  const std::uint64_t snapshot_process_allocation_bytes =
      node_compute_allocation::Bytes();
  if (!reused || storage.generation() != 3u ||
      (context.backend != Backend::Vulkan &&
       (snapshot_process_allocations != 0u ||
        snapshot_process_allocation_bytes != 0u))) {
    return 15;
  }
  // The global probe intentionally sees vendor-runtime allocation too. The
  // bounded Vulkan route/plan/barrier/hash implementation owns no dynamic
  // container, but a synchronous vkQueueSubmit may allocate inside the driver
  // (MoltenVK does on macOS). Do not encode a vendor-specific count or byte
  // value as a portable runD contract.
  node_compute_allocation::Start();
  auto second_latest = prepared.latest_device_state();
  node_compute_allocation::Stop();
  if (!second_latest || second_latest->generation() != 3u ||
      node_compute_allocation::Count() != 0u ||
      node_compute_allocation::Bytes() != 0u ||
      prepared.checkpoint_stats().device_state_acquire_count != 2u) {
    return 16;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::checkpoint
