#include "evidence.hpp"
#include "fixture.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "src/compute/virtual/run/sliding/internal.hpp"

#include <cstdio>

namespace rund_node_test_persistent_product {
namespace {

namespace accel = rund::node::accel::detail;
namespace sliding = rund::compute::detail::sliding_product_detail;

struct RunCheck final {
  enum class Stage : std::uint8_t {
    Status,
    Route,
    Queue,
    Output,
    Native,
    Authority,
    Publish,
    Stats,
    Input,
    Done,
  };

  Stage stage{Stage::Status};
  bool valid{};
};

[[nodiscard]] RunCheck check_run(const RunSample &sample) noexcept;
void report_run(const RunSample &, const RunCheck &) noexcept;

} // namespace

bool ExactPublish(const PublicationSnapshot before_primary,
                  const PublicationSnapshot after_primary,
                  const PublicationSnapshot before_alternate,
                  const PublicationSnapshot after_alternate,
                  const std::uint64_t version_before,
                  const std::uint64_t version_after,
                  const std::uint64_t recovery_after) noexcept {
  return after_primary.generation == before_primary.generation + 1u &&
         after_primary.payload_epoch == before_primary.payload_epoch + 1u &&
         after_alternate.generation == before_alternate.generation + 1u &&
         after_alternate.payload_epoch == before_alternate.payload_epoch + 1u &&
         version_after == version_before + 1u && recovery_after == 0u;
}

bool ExactRun(const RunSample &sample) noexcept {
  const RunCheck check = check_run(sample);
  if (!check.valid) {
    report_run(sample, check);
  }
  return check.valid;
}

bool ExactMode(
    const ProductRouteObservation &observation,
    const accel::PersistentResidencySlidingMode expected) noexcept {
  return observation.mode_snapshot && observation.prepared_mode == expected &&
         observation.request_mode == expected &&
         observation.control_mode == expected &&
         observation.capability_mode == expected;
}

namespace {

[[nodiscard]] RunCheck check_run(const RunSample &sample) noexcept {
  RunCheck check{};
  if (!sample.status) {
    return check;
  }
  check.stage = RunCheck::Stage::Route;
  if (!sample.observation.production_route || sample.product == nullptr ||
      sample.product->input == nullptr ||
      sample.product->input->tier() !=
          rund::compute::VirtualBackingTier::Persistent ||
      sample.product->input->max_parallel_reads() < 2u) {
    return check;
  }
  if (!ExactMode(sample.observation,
                 accel::PersistentResidencySlidingMode::BackendChunked)) {
    return check;
  }
  check.stage = RunCheck::Stage::Queue;
  const std::uint64_t submits = PhysicalSubmits(sample.coordinates);
  if (!sample.queue_read || sample.queue_after != sample.queue_before + submits) {
    return check;
  }
  check.stage = RunCheck::Stage::Output;
  if (sample.product == nullptr || !ExactOutput(*sample.product)) {
    return check;
  }
  check.stage = RunCheck::Stage::Native;
  if (!ExactNativeFinal(sample.observation, sample.coordinates)) {
    return check;
  }
  check.stage = RunCheck::Stage::Authority;
  if (!ExactAuthorityFinal(sample.observation, sample.coordinates)) {
    return check;
  }
  check.stage = RunCheck::Stage::Publish;
  if (!ExactPublish(sample.before_primary, sample.after_primary,
                    sample.before_alternate, sample.after_alternate,
                    sample.version_before, sample.version_after,
                    sample.recovery_after)) {
    return check;
  }
  check.stage = RunCheck::Stage::Stats;
  if (sample.stats.window_handoff_count != 1u ||
      sample.stats.window_batch_count != sample.coordinates ||
      sample.stats.window_queue_call_count != submits ||
      sample.command_submits != submits ||
      sample.stats.backing_read_bytes != sample.backing_input_bytes ||
      sample.stats.page_in_bytes != sample.materialized_input_bytes) {
    return check;
  }
  check.stage = RunCheck::Stage::Input;
  if (sample.input_read_calls != sample.coordinates ||
      sample.input_read_ranges != sample.pages) {
    return check;
  }
  check.stage = RunCheck::Stage::Done;
  check.valid = true;
  return check;
}

void report_run(const RunSample &sample, const RunCheck &check) noexcept {
  const auto &evidence = sample.observation.final.evidence;
  std::fprintf(
      stderr,
      "persistent product backend=%u Q=%llu stage=%u valid=%u "
      "status=%u/%u queue=%llu/%llu final=%u gpu=%llu submit=%llu chunks=%llu "
      "modes=%u/%u/%u/%u "
      "epoch=%llu host=%llu handoff=%llu batch=%llu queue_call=%llu "
      "input=%llu/%llu expected=%llu/%llu read=%llu/%llu page_in=%llu/%llu "
      "pipeline=%llu/%llu,%llu/%llu payload=%llu/%llu,%llu/%llu "
      "backing=%llu/%llu/%llu product=%u\n",
      static_cast<unsigned>(sample.backend),
      static_cast<unsigned long long>(sample.coordinates),
      static_cast<unsigned>(check.stage), static_cast<unsigned>(check.valid),
      static_cast<unsigned>(static_cast<bool>(sample.status)),
      static_cast<unsigned>(sample.status.reason()),
      static_cast<unsigned long long>(sample.queue_before),
      static_cast<unsigned long long>(sample.queue_after),
      static_cast<unsigned>(sample.observation.final_received),
      static_cast<unsigned long long>(evidence.gpu_completed_coordinates),
      static_cast<unsigned long long>(evidence.native_submit_count),
      static_cast<unsigned long long>(evidence.chunk_submit_count),
      static_cast<unsigned>(sample.observation.prepared_mode),
      static_cast<unsigned>(sample.observation.request_mode),
      static_cast<unsigned>(sample.observation.control_mode),
      static_cast<unsigned>(sample.observation.capability_mode),
      static_cast<unsigned long long>(evidence.epoch_native_submit_count),
      static_cast<unsigned long long>(evidence.host_epoch_callback_count),
      static_cast<unsigned long long>(sample.stats.window_handoff_count),
      static_cast<unsigned long long>(sample.stats.window_batch_count),
      static_cast<unsigned long long>(sample.stats.window_queue_call_count),
      static_cast<unsigned long long>(sample.input_read_calls),
      static_cast<unsigned long long>(sample.input_read_ranges),
      static_cast<unsigned long long>(sample.coordinates),
      static_cast<unsigned long long>(sample.pages),
      static_cast<unsigned long long>(sample.stats.backing_read_bytes),
      static_cast<unsigned long long>(sample.backing_input_bytes),
      static_cast<unsigned long long>(sample.stats.page_in_bytes),
      static_cast<unsigned long long>(sample.materialized_input_bytes),
      static_cast<unsigned long long>(sample.before_primary.generation),
      static_cast<unsigned long long>(sample.after_primary.generation),
      static_cast<unsigned long long>(sample.before_alternate.generation),
      static_cast<unsigned long long>(sample.after_alternate.generation),
      static_cast<unsigned long long>(sample.before_primary.payload_epoch),
      static_cast<unsigned long long>(sample.after_primary.payload_epoch),
      static_cast<unsigned long long>(sample.before_alternate.payload_epoch),
      static_cast<unsigned long long>(sample.after_alternate.payload_epoch),
      static_cast<unsigned long long>(sample.version_before),
      static_cast<unsigned long long>(sample.version_after),
      static_cast<unsigned long long>(sample.recovery_after),
      static_cast<unsigned>(sample.product != nullptr));
}

} // namespace

bool ExactNativeFinal(const ProductRouteObservation &observation,
                      const std::uint64_t coordinates) noexcept {
  if (!observation.final_received || !observation.final.check.ok ||
      observation.final.terminal != accel::NativeTerminal::Known) {
    return false;
  }
  const accel::PersistentResidencySlidingEvidence &evidence =
      observation.final.evidence;
  return evidence.coordinate_count == coordinates &&
         evidence.accepted_coordinates == coordinates &&
         evidence.gpu_completed_coordinates == coordinates &&
         evidence.completed_prefix == coordinates &&
         evidence.accepted_end == coordinates &&
         evidence.native_submit_count == PhysicalSubmits(coordinates) &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.backing_signal_count == coordinates &&
         evidence.backing_wait_count == coordinates &&
         evidence.backing_acknowledgement_count == coordinates &&
         evidence.final_callback_count == 1u &&
         evidence.queue_calls == PhysicalSubmits(coordinates) &&
         evidence.chunk_submit_count == PhysicalSubmits(coordinates);
}

bool ExactAuthorityFinal(const ProductRouteObservation &observation,
                         const std::uint64_t coordinates) noexcept {
  const auto owner =
      std::static_pointer_cast<sliding::SlidingProductOwner>(observation.owner);
  if (owner == nullptr || owner->run == nullptr ||
      owner->plan.epoch_count() != coordinates || owner->role_count != 2u) {
    return false;
  }
  rund::compute::detail::residency::execution::SlidingEvidence evidence{};
  if (!owner->run->sliding.snapshot(evidence)) {
    return false;
  }
  const auto request = owner->run->persistent_preparation.active_request();
  const accel::PersistentResidencySlidingCapability &capability =
      owner->run->persistent_preparation.backend.capability;
  const std::uint64_t expected_memory =
      sizeof(sliding::SlidingProductOwner) +
      sizeof(sliding::SlidingProductRun) +
      sizeof(rund::compute::detail::residency::execution::Plan) +
      capability.retained_bytes + capability.transient_bytes;
  const rund::storage::Usage memory = owner->memory == nullptr
                                          ? rund::storage::Usage{}
                                          : owner->memory->usage();
  return owner->run->completed && owner->run->result.status &&
         owner->memory != nullptr && owner->memory->committed() &&
         !owner->capacity && memory.physical_bytes == 0u &&
         memory.allocated_bytes == expected_memory &&
         request.coordinate_count == coordinates &&
         request.tail_local_count == 1u && request.width == 2u &&
         evidence.status && evidence.planned == coordinates &&
         evidence.admitted == coordinates &&
         evidence.terminal_frontier == coordinates &&
         evidence.persist_frontier == evidence.persist_issued &&
         !evidence.has_failure && !evidence.quarantined &&
         owner->run->sliding.quiescent();
}

bool ExactAuthorityIo(const ProductRouteObservation &observation,
                      const std::uint64_t fetch_calls,
                      const std::uint64_t fetch_hits,
                      const std::uint64_t fetch_bytes,
                      const std::uint64_t promote_bytes) noexcept {
  const auto owner =
      std::static_pointer_cast<sliding::SlidingProductOwner>(observation.owner);
  if (owner == nullptr || owner->run == nullptr) {
    return false;
  }
  rund::compute::detail::residency::execution::SlidingEvidence evidence{};
  return owner->run->sliding.snapshot(evidence) &&
         evidence.fetch_calls == fetch_calls &&
         evidence.fetch_hits == fetch_hits &&
         evidence.fetch_bytes == fetch_bytes &&
         evidence.promote_bytes == promote_bytes;
}

} // namespace rund_node_test_persistent_product

#endif
