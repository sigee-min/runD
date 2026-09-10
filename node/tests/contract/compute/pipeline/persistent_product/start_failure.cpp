#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "evidence.hpp"
#include "fixture.hpp"
#include "route.hpp"

#include "src/accel/backend/ops/table.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <string_view>
#include <utility>

namespace rund_node_test_persistent_product {
namespace {

namespace sliding = rund::compute::detail::sliding_product_detail;

[[nodiscard]] bool same_publication(const PublicationSnapshot left,
                                    const PublicationSnapshot right) noexcept {
  return left.generation == right.generation &&
         left.payload_epoch == right.payload_epoch;
}

[[nodiscard]] bool
same_live_memory(const rund::storage::Report &left,
                 const rund::storage::Report &right) noexcept {
  return left && right && left.capacity_bytes == right.capacity_bytes &&
         left.physical_bytes == right.physical_bytes &&
         left.allocated_bytes == right.allocated_bytes &&
         left.reserved_bytes == right.reserved_bytes &&
         left.available_bytes == right.available_bytes;
}

[[nodiscard]] bool
same_memory_counts(const rund::storage::Report &left,
                   const rund::storage::Report &right) noexcept {
  return left.reservation_count == right.reservation_count &&
         left.commit_count == right.commit_count &&
         left.refund_count == right.refund_count &&
         left.rejection_count == right.rejection_count;
}

[[nodiscard]] bool same_report(const rund::storage::Report &left,
                               const rund::storage::Report &right) noexcept {
  return same_live_memory(left, right) && same_memory_counts(left, right) &&
         left.peak_physical_bytes == right.peak_physical_bytes &&
         left.peak_allocated_bytes == right.peak_allocated_bytes &&
         left.peak_reserved_bytes == right.peak_reserved_bytes &&
         left.peak_used_bytes == right.peak_used_bytes;
}

[[nodiscard]] bool retained_memory(const rund::storage::Report &baseline,
                                   const rund::storage::Report &held,
                                   const std::uint64_t bytes) noexcept {
  return baseline && held && held.capacity_bytes == baseline.capacity_bytes &&
         held.physical_bytes == baseline.physical_bytes &&
         held.allocated_bytes >= baseline.allocated_bytes &&
         held.allocated_bytes - baseline.allocated_bytes == bytes &&
         held.reserved_bytes == baseline.reserved_bytes &&
         baseline.available_bytes >= bytes &&
         held.available_bytes == baseline.available_bytes - bytes &&
         held.reservation_count == baseline.reservation_count + 1u &&
         held.commit_count == baseline.commit_count + 1u &&
         held.refund_count == baseline.refund_count + 1u &&
         held.rejection_count == baseline.rejection_count;
}

[[nodiscard]] rund::node::accel::detail::PersistentResidencySlidingCapability
CapacityQuery(const std::span<const rund::node::accel::detail::
                                  PreparedResidencyPersistentSlidingRole>,
              const std::uint64_t,
              const rund::node::accel::detail::ResidencySlidingMemory memory,
              const rund::node::accel::detail::PersistentResidencySlidingMode
                  mode) noexcept {
  return {.check = {true, "ok"},
          .memory = memory,
          .width = 2u,
          .whole_run_preencoded = true,
          .host_epoch_callbacks_zero = true,
          .mode = mode};
}

[[nodiscard]] rund::node::accel::detail::PersistentResidencySlidingPreparation
CapacityPrepare(
    const rund::node::accel::detail::PersistentResidencySlidingRequest
        &request) noexcept {
  rund::node::accel::detail::PersistentResidencySlidingPreparation result{};
  result.capability.check = {false, "compute_pipeline_capacity"};
  result.capability.memory = request.memory;
  result.capability.mode = request.mode;
  return result;
}

[[nodiscard]] bool CheckPersistentCapacityReason() {
  using namespace rund::node::accel::detail;
  BackendOps ops{};
  ops.query_persistent_sliding_capability = CapacityQuery;
  ops.prepare_persistent_sliding = CapacityPrepare;

  const auto make_pipeline = [&ops] {
    auto state = std::make_shared<prepared::PipelineState>();
    state->ops = &ops;
    state->backend = std::make_shared<std::uint8_t>(1u);
    return PreparedKernelPipeline{.owner = std::move(state), .ok = true};
  };
  const std::array<PreparedResidencyPersistentSlidingRole, 2u> roles{
      PreparedResidencyPersistentSlidingRole{.pipeline = make_pipeline(),
                                             .local_count = 1u,
                                             .first_control_generation = 1u,
                                             .control_generation_stride = 1u,
                                             .first_descriptor_generation = 1u,
                                             .descriptor_generation_stride = 1u,
                                             .slot = 0u},
      PreparedResidencyPersistentSlidingRole{.pipeline = make_pipeline(),
                                             .local_count = 1u,
                                             .first_control_generation = 1u,
                                             .control_generation_stride = 1u,
                                             .first_descriptor_generation = 1u,
                                             .descriptor_generation_stride = 1u,
                                             .slot = 1u}};
  PersistentResidencySlidingRequest prototype{};
  prototype.coordinate_count = 2u;
  prototype.width = static_cast<std::uint8_t>(roles.size());
  const auto result = PrepareKernelPipelinePersistentSliding(
      prototype, std::span<const PreparedResidencyPersistentSlidingRole>{
                     roles.data(), roles.size()});
  const char *const reason = result.backend.capability.check.reason;
  return !result && !result.backend.capability.check.ok && reason != nullptr &&
         std::string_view{reason} == "compute_pipeline_capacity";
}

} // namespace

bool CheckPersistentStartFailure(const rund::compute::Backend backend,
                                 const NativeQueueCounter queue_counter,
                                 bool &unavailable) noexcept {
  constexpr std::uint64_t coordinates = 9u;
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, coordinates, prepared, unavailable)) {
    return unavailable;
  }
  const auto &device = prepared.state->pipeline->device;
  if (device == nullptr || queue_counter == nullptr) {
    return false;
  }
  const rund::storage::Report baseline =
      device->pipeline_memory_budget.report();
  const PublicationSnapshot primary_before =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_before =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  const std::uint64_t recovery_before = BackingRecovery(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (!baseline || !queue_counter(prepared.state, queue_before)) {
    return false;
  }

  sliding::inject_persistent_pipeline_begin_failure_once();
  ProductRouteObservation rejected{};
  const rund::compute::Status first =
      RunThroughPersistentProductRoute(prepared.state, rejected);
  auto rejected_owner =
      std::static_pointer_cast<sliding::SlidingProductOwner>(rejected.owner);
  std::weak_ptr<sliding::SlidingProductOwner> rejected_lifetime =
      rejected_owner;
  const rund::storage::Report held = device->pipeline_memory_budget.report();
  const auto *cached_identity = rejected_owner.get();
  const bool cached_before =
      cached_identity != nullptr &&
      prepared.state->sliding_product_cache.get() == cached_identity;
  const std::uint64_t retained_bytes =
      rejected_owner != nullptr && rejected_owner->memory != nullptr
          ? rejected_owner->memory->usage().allocated_bytes
          : 0u;
  std::uint64_t queue_rejected = 0u;
  const bool rejected_exact =
      !first && first.reason() == rund::compute::Reason::PipelineBusy &&
      rejected.production_route && !rejected.final_received &&
      rejected_owner != nullptr && rejected_owner->memory != nullptr &&
      rejected_owner->memory->committed() && !rejected_owner->capacity &&
      cached_before && retained_bytes != 0u &&
      retained_memory(baseline, held, retained_bytes) &&
      queue_counter(prepared.state, queue_rejected) &&
      queue_rejected == queue_before &&
      same_publication(primary_before,
                       SnapshotPublication(prepared.state->pipeline)) &&
      same_publication(
          alternate_before,
          SnapshotPublication(prepared.state->alternate_pipeline)) &&
      BackingVersion(*prepared.output) == version_before &&
      BackingRecovery(*prepared.output) == recovery_before;
  rejected.owner.reset();
  rejected_owner.reset();
  const auto cached_owner = prepared.state->sliding_product_cache;
  const bool cache_retained = cached_owner != nullptr &&
                              cached_owner.get() == cached_identity &&
                              !rejected_lifetime.expired();
  const rund::storage::Report warm_before =
      device->pipeline_memory_budget.report();

  ProductRouteObservation retried{};
  const rund::compute::Status second =
      RunThroughPersistentProductRoute(prepared.state, retried);
  const rund::storage::Report warm_after =
      device->pipeline_memory_budget.report();
  const PublicationSnapshot primary_after =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_after =
      SnapshotPublication(prepared.state->alternate_pipeline);
  std::uint64_t queue_after = 0u;
  const bool reused_owner = cache_retained && retried.owner != nullptr &&
                            retried.owner.get() == cached_owner.get();
  const bool warm_exact =
      same_report(held, warm_before) && same_report(warm_before, warm_after);
  const bool retry_exact =
      second && retried.production_route && retried.final_received &&
      retried.final.check.ok && reused_owner && warm_exact &&
      queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + PhysicalSubmits(coordinates) &&
      ExactMode(retried, rund::node::accel::detail::
                             PersistentResidencySlidingMode::BackendChunked) &&
      ExactOutput(prepared) && ExactNativeFinal(retried, coordinates) &&
      ExactAuthorityFinal(retried, coordinates) &&
      primary_after.generation == primary_before.generation + 1u &&
      primary_after.payload_epoch == primary_before.payload_epoch + 1u &&
      alternate_after.generation == alternate_before.generation + 1u &&
      alternate_after.payload_epoch == alternate_before.payload_epoch + 1u &&
      BackingVersion(*prepared.output) == version_before + 1u &&
      BackingRecovery(*prepared.output) == 0u;

  if (!rejected_exact || !cache_retained || !warm_exact || !retry_exact) {
    std::fprintf(
        stderr,
        "persistent start failure backend=%u first=%u/%u route=%u final=%u "
        "cache=%u/alive=%u/warm=%u/reuse=%u memory=%llu/%llu "
        "available=%llu/%llu queue=%llu/%llu/%llu publication="
        "%llu/%llu,%llu/%llu payload=%llu/%llu,%llu/%llu "
        "backing=%llu/%llu/%llu retry=%u/%u output=%u authority=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(bool(first)),
        static_cast<unsigned>(first.reason()),
        static_cast<unsigned>(rejected.production_route),
        static_cast<unsigned>(rejected.final_received),
        static_cast<unsigned>(cached_before),
        static_cast<unsigned>(cache_retained),
        static_cast<unsigned>(warm_exact), static_cast<unsigned>(reused_owner),
        static_cast<unsigned long long>(baseline.allocated_bytes),
        static_cast<unsigned long long>(held.allocated_bytes),
        static_cast<unsigned long long>(baseline.available_bytes),
        static_cast<unsigned long long>(held.available_bytes),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_rejected),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned long long>(primary_before.generation),
        static_cast<unsigned long long>(primary_after.generation),
        static_cast<unsigned long long>(alternate_before.generation),
        static_cast<unsigned long long>(alternate_after.generation),
        static_cast<unsigned long long>(primary_before.payload_epoch),
        static_cast<unsigned long long>(primary_after.payload_epoch),
        static_cast<unsigned long long>(alternate_before.payload_epoch),
        static_cast<unsigned long long>(alternate_after.payload_epoch),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(BackingVersion(*prepared.output)),
        static_cast<unsigned long long>(BackingRecovery(*prepared.output)),
        static_cast<unsigned>(bool(second)),
        static_cast<unsigned>(retried.final_received),
        static_cast<unsigned>(ExactOutput(prepared)),
        static_cast<unsigned>(ExactAuthorityFinal(retried, coordinates)));
  }
  return rejected_exact && cache_retained && warm_exact && retry_exact;
}

bool CheckPersistentTerminalUnsupported(const rund::compute::Backend backend,
                                        const NativeQueueCounter queue_counter,
                                        bool &unavailable) noexcept {
  constexpr std::uint64_t coordinates = 2u;
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, coordinates, prepared, unavailable)) {
    return unavailable;
  }
  if (prepared.state == nullptr || prepared.input == nullptr ||
      prepared.output == nullptr || queue_counter == nullptr) {
    return false;
  }
  if (!CheckPersistentCapacityReason()) {
    std::fprintf(stderr, "persistent capacity reason was not preserved through "
                         "common preparation\n");
    return false;
  }
  const PublicationSnapshot primary_before =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_before =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (!queue_counter(prepared.state, queue_before)) {
    return false;
  }

  ProductRouteObservation observation{.force_terminal_unsupported = true};
  const rund::compute::Status status =
      RunThroughPersistentProductRoute(prepared.state, observation);
  std::uint64_t queue_after = 0u;
  const bool queue_read = queue_counter(prepared.state, queue_after);
  const bool untouched =
      queue_read && queue_after == queue_before &&
      prepared.input->read_batch_calls() == 0u &&
      prepared.state->stats.pipeline.residency.epoch_count == 0u &&
      same_publication(primary_before,
                       SnapshotPublication(prepared.state->pipeline)) &&
      same_publication(
          alternate_before,
          SnapshotPublication(prepared.state->alternate_pipeline)) &&
      BackingVersion(*prepared.output) == version_before;
  const bool terminal =
      !status && status.reason() == rund::compute::Reason::BackendUnsupported &&
      observation.production_route && observation.prepared &&
      observation.owner != nullptr && !observation.final_received && untouched;
  if (!terminal) {
    std::fprintf(
        stderr,
        "persistent terminal unsupported backend=%u status=%u/%u "
        "prepared=%u owner=%u final=%u queue=%llu/%llu reads=%llu "
        "epochs=%llu version=%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(status.reason()),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.owner != nullptr),
        static_cast<unsigned>(observation.final_received),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned long long>(prepared.input->read_batch_calls()),
        static_cast<unsigned long long>(
            prepared.state->stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(BackingVersion(*prepared.output)));
  }
  return terminal;
}

} // namespace rund_node_test_persistent_product

#endif
