#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <array>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

void reserved_completion(void *, KernelResult) noexcept {}

[[nodiscard]] std::size_t
unique_sequences(const PersistentResidencySlidingRequest &request,
                 std::array<MetalSequence *, PersistentResidencySlidingCapacity>
                     &sequences) noexcept {
  std::size_t count = 0u;
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    auto *const sequence =
        static_cast<MetalSequence *>(request.roles[slot].prepared.get());
    std::size_t index = 0u;
    while (index < count && sequences[index] != sequence) {
      ++index;
    }
    if (index == count) {
      sequences[count++] = sequence;
    }
  }
  return count;
}

void cancel_claims(const std::span<MetalSequence *const> sequences) noexcept {
  for (MetalSequence *const sequence : sequences) {
    if (sequence != nullptr) {
      submission::Cancel(sequence->submission);
    }
  }
}

void reset_preaccept(Owner &owner,
                     PersistentResidencySlidingControl &control) noexcept {
  owner.pending_request = {};
  owner.rollback_request = {};
  owner.native_prepared = false;
  owner.native_pending = false;
  owner.native_retired = false;
  std::lock_guard lock{control.gate};
  if (control.active && control.accepted_end == 0u) {
    control.native.reset();
    control.active = false;
    control.service_failed = false;
    control.quarantined = false;
  }
}

[[nodiscard]] bool
ticket_ready(const Owner &owner,
             const PersistentResidencySlidingRequest &request) noexcept {
  const auto &ticket = request.ticket;
  if (ticket == nullptr || ticket->cell == nullptr ||
      ticket->active != ticket->pending.has_value()) {
    return false;
  }
  if (ticket->active) {
    return ticket->pending.has_value() &&
           persistent_sliding_request_equal(*ticket->pending, request);
  }
  return same_prepared_request(owner, request);
}

[[nodiscard]] PersistentResidencySlidingSubmitResult
out(const rund::AccelCheck status,
    const PersistentResidencySlidingSubmitEvent event =
        PersistentResidencySlidingSubmitEvent::Declined) noexcept {
  return PersistentResidencySlidingSubmitResult{status, event};
}

} // namespace

PersistentResidencySlidingSubmitResult
submit_result(const PersistentResidencySlidingRequest &request,
              PersistentResidencySlidingControl &control) noexcept {
  const std::shared_ptr<Owner> owner = owner_of(request.lowering);
  if (owner == nullptr || owner->adapter == nullptr ||
      !persistent_sliding_request_valid(owner->capability, request)) {
    return out({false, "accel_kernel_pipeline_invalid"});
  }
  if (request.mode == PersistentResidencySlidingMode::BackendChunked &&
      !persistent_sliding_chunk_valid(owner->capability, request)) {
    return out({false, "accel_kernel_pipeline_invalid"});
  }
  std::unique_lock terminal_gate{owner->adapter->residency_terminal_gate};
  std::unique_lock lock{owner->gate};
  if (!ticket_ready(*owner, request)) {
    return out({false, "accel_kernel_pipeline_invalid"});
  }
  owner->first_failure_trace = {};
  owner->last_wait_trace = {};
  owner->first_failure_trace_recorded = false;
  bool first_submit = false;
  bool final_sent = false;
  bool quarantined = false;
  {
    std::lock_guard control_lock{control.gate};
    first_submit = !control.active;
    // A clean Known close leaves historical callback evidence in Control;
    // only an active terminal may reject the next activation.  Quarantine is
    // checked separately and is never reusable.
    final_sent = control.active && control.final_callback_count != 0u;
    quarantined = control.quarantined;
  }
  if (final_sent || quarantined ||
      owner->adapter->residency_quarantined.load(std::memory_order_acquire) ||
      (first_submit && owner->command == nil) || request.coordinate_count == 0u) {
    record_first_failure(
       *owner, MetalPersistentResidencySlidingDiagnosticStage::Claims,
       request.coordinate_count,
       MetalPersistentResidencySlidingDiagnosticPredicate::State, 1u,
       PersistentSlidingNoCoordinate);
    return out({false, "compute_pipeline_busy"});
  }
  if (first_submit) {
    owner->sealed = {};
  }
  if (!first_submit) {
    std::lock_guard control_lock{control.gate};
    if (!persistent_sliding_accept_chunk_locked(owner->capability, request,
                                                control)) {
      return out({false, "accel_kernel_pipeline_invalid"});
    }
    if (!owner->native_retired) {
      return out({false, "compute_pipeline_busy"},
                 PersistentResidencySlidingSubmitEvent::Unknown);
    }
    owner->native_retired = false;
    const rund::AccelCheck encoded =
        encode_chunk(*owner, request.first_coordinate, request.chunk_count);
    if (!encoded.ok) {
      return out(encoded, PersistentResidencySlidingSubmitEvent::Unknown);
    }
    [owner->command commit];
    RecordMetalCommandSubmit(*owner->adapter);
    static_cast<void>(BeginMetalResidencyCommand(*owner->adapter));
    return out({true, "ok"},
               PersistentResidencySlidingSubmitEvent::Accepted);
  }
  if (!stage_rearm(*owner, request)) {
    return out({false, "accel_kernel_pipeline_invalid"});
  }
  std::array<MetalSequence *, PersistentResidencySlidingCapacity> sequences{};
  const std::size_t sequence_count = unique_sequences(request, sequences);
  std::size_t claimed = 0u;
  for (; claimed < sequence_count; ++claimed) {
    MetalSequence *const sequence = sequences[claimed];
    if (sequence == nullptr || sequence->adapter != owner->adapter ||
        sequence->residency_sliding.active ||
        sequence->residency_sliding.quarantined.load(
            std::memory_order_acquire) ||
        !sequence->residency_schedule.expired() ||
        !submission::Begin(sequence->submission, *sequence, reserved_completion,
                           owner.get())) {
      record_first_failure(
          *owner, MetalPersistentResidencySlidingDiagnosticStage::Claims,
          claimed, MetalPersistentResidencySlidingDiagnosticPredicate::Sequence,
          1u, PersistentSlidingNoCoordinate);
      cancel_claims(std::span<MetalSequence *const>{sequences.data(), claimed});
      abort_rearm(*owner);
      return out({false, "compute_pipeline_busy"});
    }
  }
  if (!persistent_sliding_activate_control(owner->capability, request,
                                           control)) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Control, 1u,
        PersistentSlidingNoCoordinate);
    cancel_claims(
        std::span<MetalSequence *const>{sequences.data(), sequence_count});
    abort_rearm(*owner);
    reset_preaccept(*owner, control);
    return out({false, "accel_kernel_pipeline_invalid"});
  }
  if (!prepare_rearm(*owner)) {
    cancel_claims(
        std::span<MetalSequence *const>{sequences.data(), sequence_count});
    abort_rearm(*owner);
    reset_preaccept(*owner, control);
    return out({false, "accel_kernel_pipeline_invalid"});
  }
  owner->submit_begin_ns = MonotonicNanoseconds();
  owner->run_event_base = owner->event_base;
  owner->native_retired = false;
  for (std::size_t index = 0u; index < sequence_count; ++index) {
    MetalSequence &sequence = *sequences[index];
    std::size_t role = 0u;
    while (role < request.width &&
           request.roles[role].prepared.get() != &sequence) {
      ++role;
    }
    if (role == request.width) {
      record_first_failure(
          *owner, MetalPersistentResidencySlidingDiagnosticStage::Claims, index,
          MetalPersistentResidencySlidingDiagnosticPredicate::Sequence, 2u,
          PersistentSlidingNoCoordinate);
      cancel_claims(
          std::span<MetalSequence *const>{sequences.data(), sequence_count});
      abort_rearm(*owner);
      reset_preaccept(*owner, control);
      return out({false, "accel_kernel_pipeline_invalid"});
    }
    sequence.residency_schedule = owner;
    sequence.residency_sliding.active = true;
    sequence.residency_sliding.active_owner = request.roles[role].prepared;
  }
  const rund::AccelCheck encoded =
      encode_chunk(*owner, request.first_coordinate,
                   request.mode == PersistentResidencySlidingMode::OneSubmit
                       ? request.coordinate_count
                       : request.chunk_count);
  if (!encoded.ok) {
    cancel_claims(
        std::span<MetalSequence *const>{sequences.data(), sequence_count});
    abort_rearm(*owner);
    reset_preaccept(*owner, control);
    return out(encoded);
  }
  [owner->command commit];
  const PersistentResidencySlidingRequest pending = owner->pending_request;
  const bool committed = commit_rearm(*owner);
  if (!committed || !same_prepared_request(*owner, request)) {
    record_first_failure(
        *owner,
        committed
            ? MetalPersistentResidencySlidingDiagnosticStage::QueueCommit
            : MetalPersistentResidencySlidingDiagnosticStage::CommonCommit,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Commit,
        committed ? 2u : 1u, PersistentSlidingNoCoordinate);
    owner->prepared = committed ? request : pending;
    owner->prepared.lowering.reset();
    owner->owner_nonce = request.owner_nonce;
    owner->quarantine_ticket = true;
    persistent_sliding_quarantine_ticket(request.ticket);
    quarantine_owner(*owner);
    return out({false, "compute_device_lost"},
               PersistentResidencySlidingSubmitEvent::AcceptedUnknown);
  }
  RecordMetalCommandSubmit(*owner->adapter);
  static_cast<void>(BeginMetalResidencyCommand(*owner->adapter));
  return out({true, "ok"},
              PersistentResidencySlidingSubmitEvent::Accepted);
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

const PersistentResidencySlidingServiceOps &
MetalPersistentResidencySlidingServiceOps() noexcept {
  static const PersistentResidencySlidingServiceOps ops{
      .submit_result = metal_persistent_sliding::submit_result,
      .wait_done = metal_persistent_sliding::wait_done,
      .signal_ready = metal_persistent_sliding::signal_ready,
      .ack_done = metal_persistent_sliding::acknowledge_done,
      .fail_service = metal_persistent_sliding::fail_service,
      .quarantine_unknown = metal_persistent_sliding::quarantine_unknown,
  };
  return ops;
}

#endif

} // namespace rund::node::accel::detail
