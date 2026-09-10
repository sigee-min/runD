#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <array>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

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

} // namespace

Binding bind(PersistentResidencySlidingControl &control) noexcept {
  Binding binding{};
  {
    std::lock_guard lock{control.gate};
    binding.native = control.native;
    binding.plan_identity = control.plan_identity;
    binding.token = control.token;
    binding.generation = control.generation;
    binding.owner_nonce = control.owner_nonce;
  }
  binding.owner = owner_of(binding.native);
  return binding;
}

bool valid_binding(const Binding &binding, const Owner &owner,
                   PersistentResidencySlidingControl &control) noexcept {
  std::lock_guard lock{control.gate};
  return binding.owner != nullptr && binding.owner.get() == &owner &&
         binding.native != nullptr && control.native == binding.native &&
         control.plan_identity == binding.plan_identity &&
         control.token == binding.token &&
         control.generation == binding.generation &&
         owner.magic == OwnerMagic && owner.owner_nonce == binding.owner_nonce &&
         owner.prepared.plan_identity ==
                                         binding.plan_identity &&
         owner.prepared.token == binding.token &&
         owner.prepared.generation == binding.generation &&
         owner.prepared.owner_nonce == binding.owner_nonce;
}

void release_known_claims(Owner &owner) noexcept {
  std::array<MetalSequence *, PersistentResidencySlidingCapacity> sequences{};
  const std::size_t count = unique_sequences(owner.prepared, sequences);
  for (std::size_t index = 0u; index < count; ++index) {
    MetalSequence &sequence = *sequences[index];
    sequence.residency_schedule.reset();
    sequence.residency_sliding.active = false;
    sequence.residency_sliding.active_owner.reset();
    submission::Cancel(sequence.submission);
  }
}

void quarantine_unknown(Owner &owner,
                        PersistentResidencySlidingControl &control) noexcept {
  quarantine_owner(owner);
  {
    std::lock_guard control_lock{control.gate};
    // Unknown terminal ownership is retained in native, while the active
    // admission bit is closed.  The immutable accepted-prefix counters stay
    // available to the common terminal callback and recovery owner.
    control.active = false;
    control.quarantined = true;
  }
}

void quarantine_owner(Owner &owner) noexcept {
  if (owner.adapter != nullptr) {
    owner.adapter->residency_quarantined.store(true, std::memory_order_release);
    SetMetalLastError(*owner.adapter, "compute_device_lost");
  }
  std::array<MetalSequence *, PersistentResidencySlidingCapacity> sequences{};
  const std::size_t count = unique_sequences(owner.prepared, sequences);
  for (std::size_t index = 0u; index < count; ++index) {
    MetalSequence &sequence = *sequences[index];
    std::size_t role = 0u;
    while (role < owner.prepared.width &&
           owner.prepared.roles[role].prepared.get() != &sequence) {
      ++role;
    }
    sequence.residency_sliding.quarantined.store(true,
                                                 std::memory_order_release);
    if (role < owner.prepared.width) {
      sequence.residency_sliding.quarantine_owner =
          owner.prepared.roles[role].prepared;
    }
    sequence.residency_submission.phase =
        MetalResidencySubmissionPhase::Quarantined;
  }
  owner.quarantine = owner.shared_from_this();
}

void quarantine_unknown(PersistentResidencySlidingControl &control) noexcept {
  const Binding binding = bind(control);
  if (binding.owner != nullptr) {
    std::lock_guard owner_lock{binding.owner->gate};
    if (!valid_binding(binding, *binding.owner, control)) {
      quarantine_owner(*binding.owner);
      std::lock_guard control_lock{control.gate};
      control.active = false;
      control.quarantined = true;
      return;
    }
    quarantine_unknown(*binding.owner, control);
    return;
  }
  std::lock_guard control_lock{control.gate};
  control.active = false;
  control.quarantined = true;
}

PersistentResidencySlidingFinal
make_final(Owner &owner, const PersistentResidencySlidingControl &control,
           const rund::AccelCheck check, const NativeTerminal terminal,
           const std::uint64_t first_failure) noexcept {
  PersistentResidencySlidingFinal result{};
  MetalPersistentResidencySlidingDiagnostics sealed{};
  {
    std::lock_guard lock{control.gate};
    result = PersistentResidencySlidingFinal{
        .check = check,
        .terminal = terminal,
        .evidence =
            PersistentResidencySlidingEvidence{
                .plan_identity = owner.prepared.plan_identity,
                .token = owner.prepared.token,
                .generation = owner.prepared.generation,
                .owner_nonce = owner.prepared.owner_nonce,
                .cell_id = owner.prepared.cell_id,
                .cell_domain = owner.prepared.cell_domain,
                .coordinate_count = owner.prepared.coordinate_count,
                .accepted_coordinates = control.accepted_end,
                .gpu_completed_coordinates =
                    control.gpu_completed_coordinates,
                .completed_prefix = control.completed_prefix,
                .suppressed_first = control.suppressed_first,
                .suppressed_count = control.suppressed_count,
                .native_submit_count = control.native_submit_count,
                .epoch_native_submit_count =
                    control.epoch_native_submit_count,
                .backend_epoch_callback_count =
                    control.backend_epoch_callback_count,
                .host_epoch_callback_count = 0u,
                .backing_wait_count = control.backing_wait_count,
                .backing_signal_count = control.backing_signal_count,
                .backing_acknowledgement_count =
                    control.backing_acknowledgement_count,
                .backing_service_failure_count =
                    control.backing_service_failure_count,
                .failed_admission_count = control.failed_admission_count,
                .first_failed_admission_coordinate =
                    control.first_failed_admission_coordinate,
                .first_service_failure_coordinate =
                    control.first_service_failure_coordinate,
                .final_callback_count = control.final_callback_count,
                .queue_calls = control.queue_calls,
                .accepted_end = control.accepted_end,
                .chunk_submit_count = control.chunk_submit_count,
                .first_failure_coordinate = first_failure,
                .completed_ns = MonotonicNanoseconds() - owner.submit_begin_ns,
                .width = owner.prepared.width,
            },
    };
    sealed.encoded_coordinate_count = owner.prepared.coordinate_count;
    sealed.encoded_intermediate_bytes = owner.encoded_intermediate_bytes;
    sealed.native_submit_count = control.native_submit_count;
    sealed.epoch_native_submit_count = control.epoch_native_submit_count;
    sealed.backend_epoch_callback_count =
        control.backend_epoch_callback_count;
    sealed.queue_commit_count = control.queue_calls;
    sealed.final_callback_count = control.final_callback_count;
    sealed.submitted = terminal == NativeTerminal::UnknownMayWrite;
    sealed.final_sent = sealed.submitted &&
                        control.final_callback_count != 0u;
    sealed.quarantined = sealed.submitted || control.quarantined;
  }
  sealed.first_failure = owner.first_failure_trace;
  sealed.last_wait = owner.last_wait_trace;
  owner.sealed.value = sealed;
  owner.sealed.valid = true;
  return result;
}

void deliver_final(Owner &owner, PersistentResidencySlidingFinal &&final) noexcept {
  PersistentResidencySlidingFinalCompletion completion = nullptr;
  void *user = nullptr;
  {
    std::lock_guard lock{owner.gate};
    completion = owner.prepared.final;
    user = owner.prepared.user;
    owner.prepared.final = nullptr;
    owner.prepared.user = nullptr;
  }
  const bool quarantined =
      final.terminal == NativeTerminal::UnknownMayWrite;
  // Unknown owners deliberately self-retain after this call.  Remove the raw
  // callback target before invoking user code so a late native terminal can
  // never rediscover caller-owned storage after that caller has returned.
  if (completion != nullptr && user != nullptr) {
    completion(user, std::move(final));
  }
  if (!quarantined) {
    std::lock_guard lock{owner.gate};
    owner.prepared = {};
    owner.native_retired = false;
  }
}

rund::AccelCheck acknowledge_done(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingAcknowledgeDone &acknowledgement) noexcept {
  const Binding binding = bind(control);
  const std::shared_ptr<Owner> owner = binding.owner;
  if (owner == nullptr || owner->adapter == nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  PersistentResidencySlidingFinal final{};
  bool emit = false;
  {
    std::unique_lock terminal_gate{owner->adapter->residency_terminal_gate};
    std::unique_lock lock{owner->gate};
    bool closed = false;
    {
      std::lock_guard control_lock{control.gate};
      closed = !control.active || control.final_callback_count != 0u ||
               control.quarantined;
    }
    if (!valid_binding(binding, *owner, control) || closed ||
        !persistent_sliding_accept_done_acknowledgement(control,
                                                        acknowledgement)) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    const bool success_ready = persistent_sliding_final_ready(control);
    bool failure_ready = false;
    {
      std::lock_guard control_lock{control.gate};
      failure_ready =
          control.active &&
          (control.service_failed || control.failed_admission_count != 0u) &&
          !control.quarantined &&
          control.native_submit_count == control.queue_calls &&
          control.queue_calls == control.chunk_submit_count &&
          control.native_submit_count <=
              (control.mode == PersistentResidencySlidingMode::OneSubmit
                   ? 1u
                   : persistent_sliding_chunk_count(control.coordinate_count)) &&
          control.epoch_native_submit_count == 0u &&
          control.backend_epoch_callback_count == 0u &&
          control.next_acknowledgement_coordinate ==
              control.accepted_end &&
          control.backing_acknowledgement_count ==
              control.accepted_end;
    }
    if (success_ready || failure_ready) {
      const std::uint64_t end = failure_ready
                                    ? control.accepted_end
                                    : owner->prepared.coordinate_count;
      if (!owner->native_retired ||
          control.gpu_completed_coordinates != end) {
        return {false, "compute_device_lost"};
      }
      {
        std::lock_guard control_lock{control.gate};
        control.final_callback_count = 1u;
      }
      release_known_claims(*owner);
      final = make_final(*owner, control,
                         failure_ready ? control.first_failure
                                       : rund::AccelCheck{true, "ok"},
                         NativeTerminal::Known,
                         failure_ready ? control.first_failure_coordinate
                                       : PersistentSlidingNoCoordinate);
      {
        std::lock_guard control_lock{control.gate};
        control.active = false;
        control.native.reset();
      }
      emit = true;
    }
  }
  if (emit) {
    deliver_final(*owner, std::move(final));
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
