#include "local.hpp"

#include "../../target/selection.hpp"
#include "../allocation.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace rund_node_test_pipeline {
namespace {

[[nodiscard]] bool
SameCheckpointStats(const rund::compute::CheckpointStats &left,
                    const rund::compute::CheckpointStats &right) noexcept {
  return left.device_state_acquire_count == right.device_state_acquire_count &&
         left.device_state_rebase_count == right.device_state_rebase_count &&
         left.device_state_copy_byte_count ==
             right.device_state_copy_byte_count &&
         left.device_state_copy_command_count ==
             right.device_state_copy_command_count &&
         left.reusable_snapshot_count == right.reusable_snapshot_count &&
         left.reusable_snapshot_byte_count ==
             right.reusable_snapshot_byte_count &&
         left.reusable_snapshot_hash == right.reusable_snapshot_hash &&
         left.reusable_snapshot_transfer_count ==
             right.reusable_snapshot_transfer_count;
}

} // namespace

[[nodiscard]] int CheckReusableCheckpoints(rund::compute::Device &device,
                                           const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> initial{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> thrice{4, 5, 6, 7};
  constexpr std::array<std::int32_t, 4u> fourth{5, 6, 7, 8};

  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-reusable-checkpoint", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = Upload(device, initial);
  auto second = device.buffer<std::int32_t>(initial.size());
  if (!advance || !first || !second) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .state(*first, *second)
                      .then(*advance, read(*first), write(*second))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 2;
  }
  const std::shared_ptr<detail::PipelineState> source_state =
      detail::PipelineStateAccess::state(*prepared);
  if (source_state == nullptr || source_state->publication == nullptr) {
    return 3;
  }

  auto storage_result = prepared->snapshot_storage();
  auto small_result = prepared->snapshot_storage(sizeof(initial) - 1u);
  auto latest_result = prepared->latest_device_state();
  if (!storage_result || !small_result || !latest_result) {
    return 4;
  }
  SnapshotStorage storage = std::move(storage_result).value();
  SnapshotStorage small = std::move(small_result).value();
  LatestDeviceState latest = std::move(latest_result).value();
  if (!storage.valid() || storage.has_snapshot() ||
      storage.capacity() != sizeof(initial) || storage.field_capacity() != 1u ||
      storage.generation() != 0u || storage.hash() != 0u ||
      storage.fingerprint() || !small.valid() || small.has_snapshot() ||
      !latest.valid() || latest.generation() != 0u ||
      latest.fingerprint() != prepared->fingerprint() ||
      prepared->checkpoint_stats().device_state_acquire_count != 1u) {
    return 5;
  }
  const Status empty_restore = prepared->restore(storage);
  const Status too_small = prepared->snapshot_into(small);
  if (empty_restore || empty_restore.reason() != Reason::PipelineInvalid ||
      prepared->poisoned() || prepared->generation() != 0u || too_small ||
      too_small.reason() != Reason::BufferCapacity || small.has_snapshot()) {
    return 6;
  }
  if (!prepared->snapshot_into(storage) || !storage.has_snapshot() ||
      storage.generation() != 0u || storage.hash() == 0u ||
      storage.fingerprint() != prepared->fingerprint()) {
    return 7;
  }
  const auto initial_snapshot = prepared->snapshot();
  if (!initial_snapshot || initial_snapshot->generation() != 0u ||
      initial_snapshot->hash() != storage.hash() ||
      initial_snapshot->fingerprint() != storage.fingerprint()) {
    return 8;
  }

  // A cold parity-zero handoff adopts the one publication authority without
  // touching payload bytes.
  auto parity_zero = pipeline(device)
                         .state(*first, *second)
                         .then(*advance, read(*first), write(*second))
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
  if (!prepared->run() || prepared->generation() != 1u ||
      latest.generation() != 1u) {
    return 10;
  }

  // Parity one must seed the alternate native generation stream as the next
  // selected stream. This catches a generation-stride-two handoff bug.
  auto parity_one = pipeline(device)
                        .state(*first, *second)
                        .then(*advance, read(*first), write(*second))
                        .restore(latest)
                        .commit()
                        .prepare();
  const std::shared_ptr<detail::PipelineState> parity_one_state =
      parity_one ? detail::PipelineStateAccess::state(*parity_one)
                 : std::shared_ptr<detail::PipelineState>{};
  if (!parity_one || parity_one_state == nullptr ||
      parity_one_state->publication != source_state->publication ||
      parity_one->generation() != 1u || !parity_one->run() ||
      latest.generation() != 2u || prepared->generation() != 2u ||
      !parity_zero->run() || latest.generation() != 3u) {
    return 11;
  }
  std::array<std::int32_t, initial.size()> observed{};
  if (!ReadExact(*prepared, *second, observed) || observed != thrice) {
    return 12;
  }

  if (!prepared->snapshot_into(storage) || storage.generation() != 3u) {
    return 13;
  }
  const auto current_snapshot = prepared->snapshot();
  if (!current_snapshot || current_snapshot->generation() != 3u ||
      current_snapshot->hash() != storage.hash() ||
      current_snapshot->fingerprint() != storage.fingerprint()) {
    return 14;
  }

  // Resetting a shared publication from odd parity to parity zero at the same
  // generation makes every sibling wrapper's native selector stale. The next
  // run must compare both generation and parity before choosing its prepared
  // stream; generation alone would advance the stride-two stream to g + 2.
  auto parity_probe_result = prepared->snapshot_storage();
  if (!parity_probe_result) {
    return 14;
  }
  SnapshotStorage parity_probe = std::move(parity_probe_result).value();
  if (!prepared->restore(storage) || source_state->publication->parity != 0u ||
      parity_zero_state->native_generation != 3u ||
      parity_zero_state->native_parity != 1u || !parity_zero->run() ||
      parity_zero->generation() != 4u ||
      !parity_zero->snapshot_into(parity_probe) ||
      parity_probe.generation() != 4u ||
      parity_probe.hash() == storage.hash() || !prepared->restore(storage) ||
      prepared->generation() != 3u || source_state->publication->parity != 0u ||
      !ReadExact(*parity_zero, *first, observed) || observed != thrice ||
      parity_zero->stats().output_hash != 0u ||
      !prepared->restore(parity_probe) || prepared->generation() != 4u ||
      !ReadExact(*parity_zero, *first, observed) || observed != fourth ||
      parity_zero->stats().output_hash != 0u || !prepared->restore(storage) ||
      prepared->generation() != 3u) {
    return 14;
  }
  node_compute_allocation::Start();
  const Status reused = prepared->snapshot_into(storage);
  node_compute_allocation::Stop();
  const std::uint64_t snapshot_process_allocations =
      node_compute_allocation::Count();
  const std::uint64_t snapshot_process_allocation_bytes =
      node_compute_allocation::Bytes();
  if (!reused || storage.generation() != 3u ||
      (backend != Backend::Vulkan &&
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
  auto second_latest = prepared->latest_device_state();
  node_compute_allocation::Stop();
  if (!second_latest || second_latest->generation() != 3u ||
      node_compute_allocation::Count() != 0u ||
      node_compute_allocation::Bytes() != 0u ||
      prepared->checkpoint_stats().device_state_acquire_count != 2u) {
    return 16;
  }

  // One reservation belongs to the shared publication authority, not to one
  // Pipeline wrapper. Busy operations preserve the active storage bank.
  const std::uint64_t busy_generation = storage.generation();
  const std::uint64_t busy_hash = storage.hash();
  if (!detail::queue_pipeline(source_state)) {
    return 17;
  }
  const Status busy_snapshot = prepared->snapshot_into(storage);
  const Status busy_peer = parity_one->run();
  const Status busy_restore = prepared->restore(storage);
  const Status cancelled = detail::cancel_pipeline(source_state);
  if (busy_snapshot || busy_snapshot.reason() != Reason::PipelineBusy ||
      busy_peer || busy_peer.reason() != Reason::PipelineBusy || busy_restore ||
      busy_restore.reason() != Reason::PipelineBusy || cancelled ||
      cancelled.reason() != Reason::Cancelled ||
      storage.generation() != busy_generation || storage.hash() != busy_hash ||
      latest.generation() != 3u || prepared->poisoned()) {
    return 18;
  }

  // Compatible disjoint owners take exactly one selected field, B bytes, to
  // destination parity zero without any host upload/download accounting.
  auto copy_first = device.buffer<std::int32_t>(initial.size());
  auto copy_second = device.buffer<std::int32_t>(initial.size());
  auto copied =
      copy_first && copy_second
          ? pipeline(device)
                .state(*copy_first, *copy_second)
                .then(*advance, read(*copy_first), write(*copy_second))
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
      backend == Backend::Cpu ? 0u : 1u;
  if (!copied_latest || copied->generation() != 3u || copied->poisoned() ||
      copied_stats.device_state_copy_byte_count != sizeof(initial) ||
      copied_stats.device_state_copy_command_count != expected_copy_commands ||
      after_copy.uploaded_bytes != before_copy.uploaded_bytes ||
      after_copy.downloaded_bytes != before_copy.downloaded_bytes ||
      after_copy.download_events != before_copy.download_events ||
      !ReadExact(*copied, *copy_first, observed) || observed != thrice ||
      !copied->run() || copied->generation() != 4u ||
      !ReadExact(*copied, *copy_second, observed) || observed != fourth) {
    return 20;
  }

  // An exposed duplicate selector and every partial/reversed alias are
  // rejected before mutation. The existing destination stays usable.
  auto stranded = pipeline(device)
                      .state(*first, *second)
                      .then(*advance, read(*first), write(*second))
                      .commit()
                      .prepare();
  const std::shared_ptr<detail::PipelineState> stranded_state =
      stranded ? detail::PipelineStateAccess::state(*stranded)
               : std::shared_ptr<detail::PipelineState>{};
  const std::shared_ptr<detail::PipelinePublicationState> stranded_publication =
      stranded_state == nullptr ? nullptr : stranded_state->publication;
  const Status stranded_restore = stranded
                                      ? stranded->restore(latest)
                                      : Status::fail(Reason::PipelineInvalid);
  if (!stranded || stranded_restore ||
      stranded_restore.reason() != Reason::PipelineInvalid ||
      stranded->poisoned() || stranded->generation() != 0u ||
      stranded_state->publication != stranded_publication) {
    return 21;
  }
  auto partial_pending = device.buffer<std::int32_t>(initial.size());
  auto partial =
      partial_pending
          ? pipeline(device)
                .state(*first, *partial_pending)
                .then(*advance, read(*first), write(*partial_pending))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> partial_state =
      partial ? detail::PipelineStateAccess::state(*partial)
              : std::shared_ptr<detail::PipelineState>{};
  const std::shared_ptr<detail::PipelinePublicationState> partial_publication =
      partial_state == nullptr ? nullptr : partial_state->publication;
  const Status partial_restore = partial
                                     ? partial->restore(latest)
                                     : Status::fail(Reason::PipelineInvalid);
  if (!partial || partial_restore ||
      partial_restore.reason() != Reason::BindingDuplicate ||
      partial->poisoned() || partial->generation() != 0u ||
      partial_state->publication != partial_publication) {
    return 22;
  }
  auto reversed = pipeline(device)
                      .state(*second, *first)
                      .then(*advance, read(*second), write(*first))
                      .commit()
                      .prepare();
  const Status reversed_restore = reversed
                                      ? reversed->restore(latest)
                                      : Status::fail(Reason::PipelineInvalid);
  if (!reversed || reversed_restore ||
      reversed_restore.reason() != Reason::BindingDuplicate ||
      reversed->poisoned() || reversed->generation() != 0u) {
    return 23;
  }

  // A previously valid bank survives both byte-capacity and field-capacity
  // failures byte-for-byte.
  constexpr std::array<std::int32_t, 2u> tiny_initial{9, 10};
  auto tiny_advance =
      on(device)
          .map<std::int32_t>("pipeline-reusable-checkpoint-tiny",
                             tiny_initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto tiny_first = Upload(device, tiny_initial);
  auto tiny_second = device.buffer<std::int32_t>(tiny_initial.size());
  auto tiny =
      tiny_advance && tiny_first && tiny_second
          ? pipeline(device)
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
  const CheckpointStats before_capacity_failure = prepared->checkpoint_stats();

  auto pair_first = Upload(device, initial);
  auto pair_second = device.buffer<std::int32_t>(initial.size());
  auto pair_other_first = Upload(device, initial);
  auto pair_other_second = device.buffer<std::int32_t>(initial.size());
  auto paired =
      pair_first && pair_second && pair_other_first && pair_other_second
          ? pipeline(device)
                .state(*pair_first, *pair_second)
                .state(*pair_other_first, *pair_other_second)
                .then(*advance, read(*pair_first), write(*pair_second))
                .then(*advance, read(*pair_other_first),
                      write(*pair_other_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const Status byte_failure = prepared->snapshot_into(small);
  const Status field_failure = paired ? paired->snapshot_into(roomy)
                                      : Status::fail(Reason::PipelineInvalid);
  if (!paired || byte_failure ||
      byte_failure.reason() != Reason::BufferCapacity || field_failure ||
      field_failure.reason() != Reason::BufferCapacity ||
      !roomy.has_snapshot() || roomy.generation() != retained_generation ||
      roomy.hash() != retained_hash ||
      roomy.fingerprint() != retained_fingerprint ||
      !SameCheckpointStats(before_capacity_failure,
                           prepared->checkpoint_stats())) {
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
      paired_storage.capacity() != sizeof(initial) * 2u ||
      paired_storage.hash() != paired_snapshot->hash()) {
    return 28;
  }

  // Portable storage crosses Device ownership; the live resident handle does
  // not. A different graph with the same byte shape is rejected unchanged.
  auto replacement = open(rund::node::test_contract::target_for(backend, 2u));
  if (!replacement) {
    return 29;
  }
  auto replacement_advance =
      on(*replacement)
          .map<std::int32_t>("pipeline-reusable-checkpoint", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto replacement_first = replacement->buffer<std::int32_t>(initial.size());
  auto replacement_second = replacement->buffer<std::int32_t>(initial.size());
  auto portable =
      replacement_advance && replacement_first && replacement_second
          ? pipeline(*replacement)
                .state(*replacement_first, *replacement_second)
                .then(*replacement_advance, read(*replacement_first),
                      write(*replacement_second))
                .restore(storage)
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!portable || portable->generation() != 3u ||
      !ReadExact(*portable, *replacement_first, observed) ||
      observed != thrice || !portable->run() || portable->generation() != 4u ||
      !ReadExact(*portable, *replacement_second, observed) ||
      observed != fourth) {
    return 30;
  }

  // Host-owned storage is backend-independent. When this test target exposes
  // another backend, restore the exact bytes/hash into it and prove execution
  // parity. The live resident handle must reject that distinct Device/backend
  // without mutating the restored Pipeline.
  Backend portable_backend = Backend::Unavailable;
  for (const Backend candidate :
       rund::node::test_contract::selected_compute_backends()) {
    if (candidate != backend) {
      portable_backend = candidate;
      break;
    }
  }
  if (portable_backend != Backend::Unavailable) {
    auto cross_backend_device =
        open(rund::node::test_contract::target_for(portable_backend, 2u));
    if (!cross_backend_device) {
      return 38;
    }
    auto cross_backend_advance =
        on(*cross_backend_device)
            .map<std::int32_t>("pipeline-reusable-checkpoint", initial.size(),
                               [](auto value) { return value + 1; })
            .compile();
    auto cross_backend_first =
        cross_backend_device->buffer<std::int32_t>(initial.size());
    auto cross_backend_second =
        cross_backend_device->buffer<std::int32_t>(initial.size());
    auto cross_backend =
        cross_backend_advance && cross_backend_first && cross_backend_second
            ? pipeline(*cross_backend_device)
                  .state(*cross_backend_first, *cross_backend_second)
                  .then(*cross_backend_advance, read(*cross_backend_first),
                        write(*cross_backend_second))
                  .restore(storage)
                  .commit()
                  .prepare()
            : Result<Pipeline>::fail(Reason::PipelineInvalid);
    if (!cross_backend || cross_backend->generation() != 3u ||
        !ReadExact(*cross_backend, *cross_backend_first, observed) ||
        observed != thrice || !cross_backend->run() ||
        cross_backend->generation() != 4u ||
        !ReadExact(*cross_backend, *cross_backend_second, observed) ||
        observed != fourth) {
      return 38;
    }
    const Status resident_backend_mismatch = cross_backend->restore(latest);
    if (resident_backend_mismatch ||
        resident_backend_mismatch.reason() != Reason::BindingDeviceMismatch ||
        cross_backend->generation() != 4u || cross_backend->poisoned() ||
        !ReadExact(*cross_backend, *cross_backend_second, observed) ||
        observed != fourth) {
      return 38;
    }
  }
  auto resident_first = Upload(*replacement, initial);
  auto resident_second = replacement->buffer<std::int32_t>(initial.size());
  auto cross_device =
      replacement_advance && resident_first && resident_second
          ? pipeline(*replacement)
                .state(*resident_first, *resident_second)
                .then(*replacement_advance, read(*resident_first),
                      write(*resident_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> cross_device_state =
      cross_device ? detail::PipelineStateAccess::state(*cross_device)
                   : std::shared_ptr<detail::PipelineState>{};
  const std::shared_ptr<detail::PipelinePublicationState>
      cross_device_publication =
          cross_device_state == nullptr ? nullptr
                                        : cross_device_state->publication;
  const Status cross_device_restore =
      cross_device ? cross_device->restore(latest)
                   : Status::fail(Reason::PipelineInvalid);
  if (!cross_device || cross_device_restore ||
      cross_device_restore.reason() != Reason::BindingDeviceMismatch ||
      cross_device->poisoned() || cross_device->generation() != 0u ||
      cross_device_state->publication != cross_device_publication ||
      !ReadExact(*cross_device, *resident_first, observed) ||
      observed != initial) {
    return 31;
  }

  auto incompatible_program =
      on(device)
          .map<std::int32_t>("pipeline-reusable-incompatible", initial.size(),
                             [](auto value) { return value + 2; })
          .compile();
  auto incompatible_first = Upload(device, initial);
  auto incompatible_second = device.buffer<std::int32_t>(initial.size());
  auto incompatible =
      incompatible_program && incompatible_first && incompatible_second
          ? pipeline(device)
                .state(*incompatible_first, *incompatible_second)
                .then(*incompatible_program, read(*incompatible_first),
                      write(*incompatible_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const Status incompatible_restore =
      incompatible ? incompatible->restore(storage)
                   : Status::fail(Reason::PipelineInvalid);
  if (!incompatible || incompatible_restore ||
      incompatible_restore.reason() != Reason::PipelineInvalid ||
      incompatible->poisoned() || incompatible->generation() != 0u ||
      !ReadExact(*incompatible, *incompatible_first, observed) ||
      observed != initial) {
    return 32;
  }

  // A non-state Pipeline has no resident checkpoint authority.
  auto ordinary_input = Upload(device, initial);
  auto ordinary_output = device.buffer<std::int32_t>(initial.size());
  auto ordinary =
      ordinary_input && ordinary_output
          ? pipeline(device)
                .then(*advance, read(*ordinary_input), write(*ordinary_output))
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
  auto capacity_first = Upload(device, initial);
  auto capacity_second = device.buffer<std::int32_t>(initial.size());
  auto capacity =
      capacity_first && capacity_second
          ? pipeline(device)
                .state(*capacity_first, *capacity_second)
                .then(*advance, read(*capacity_first), write(*capacity_second))
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
      !ReadExact(*capacity, *capacity_first, observed) || observed != initial) {
    return 35;
  }

  // Observation identity must never become ambiguous through payload-epoch
  // wrap. Every payload-changing route fails before claims or submission at
  // UINT64_MAX, while the exact same-authority zero-byte rebase remains valid.
  auto epoch_first = Upload(device, initial);
  auto epoch_second = device.buffer<std::int32_t>(initial.size());
  auto epoch =
      epoch_first && epoch_second
          ? pipeline(device)
                .state(*epoch_first, *epoch_second)
                .then(*advance, read(*epoch_first), write(*epoch_second))
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
  const Status epoch_host = epoch->restore(storage);
  const Status epoch_copy = epoch->restore(latest);
  const Status epoch_run = epoch->run();
  if (!epoch_noop || epoch_host ||
      epoch_host.reason() != Reason::PipelineCapacity || epoch_copy ||
      epoch_copy.reason() != Reason::PipelineCapacity || epoch_run ||
      epoch_run.reason() != Reason::PipelineCapacity || epoch->poisoned() ||
      epoch->generation() != 0u || epoch->stats().command_submits != 0u ||
      epoch_state->publication->payload_epoch !=
          std::numeric_limits<std::uint64_t>::max() ||
      !ReadExact(*epoch, *epoch_first, observed) || observed != initial) {
    return 36;
  }

  // Exercise enough alternating commits to cross the native parity streams
  // repeatedly. The live selector and both reusable-bank identities must
  // follow every publication, not merely the first hand-off.
  constexpr std::uint64_t many_commits = 65u;
  const CheckpointStats many_before = prepared->checkpoint_stats();
  std::uint64_t previous_hash = storage.hash();
  for (std::uint64_t commit = 1u; commit <= many_commits; ++commit) {
    if (!prepared->run() || !prepared->snapshot_into(storage) ||
        prepared->generation() != 3u + commit ||
        latest.generation() != 3u + commit ||
        storage.generation() != 3u + commit || storage.hash() == 0u ||
        storage.hash() == previous_hash) {
      return 37;
    }
    previous_hash = storage.hash();
  }
  const CheckpointStats many_after = prepared->checkpoint_stats();
  std::array<std::int32_t, initial.size()> many_published{};
  for (std::size_t index = 0u; index < initial.size(); ++index) {
    many_published[index] =
        initial[index] + 3 + static_cast<std::int32_t>(many_commits);
  }
  // Public transactional reads intentionally project either declared owner
  // onto the current publication selector; the inactive bank is not a
  // separately observable Pipeline result.
  if (many_after.reusable_snapshot_count -
              many_before.reusable_snapshot_count !=
          many_commits ||
      many_after.reusable_snapshot_byte_count -
              many_before.reusable_snapshot_byte_count !=
          many_commits * sizeof(initial) ||
      many_after.reusable_snapshot_transfer_count -
              many_before.reusable_snapshot_transfer_count !=
          (backend == Backend::Cpu ? 0u : many_commits) ||
      !ReadExact(*prepared, *second, observed) || observed != many_published ||
      !ReadExact(*prepared, *first, observed) || observed != many_published) {
    return 37;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
