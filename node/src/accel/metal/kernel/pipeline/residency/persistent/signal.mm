#include "internal.hpp"

#include "../sliding/internal.hpp"

#include <atomic>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
signal_ready(PersistentResidencySlidingControl &control,
             const PersistentResidencySlidingReadySignal &signal) noexcept {
  const Binding binding = bind(control);
  const std::shared_ptr<Owner> owner = binding.owner;
  if (owner == nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  std::lock_guard lock{owner->gate};
  if (!valid_binding(binding, *owner, control) ||
      signal.identity.coordinate >= owner->prepared.coordinate_count) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 1u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  bool active = false;
  bool final_sent = false;
  bool quarantined = false;
  {
    std::lock_guard control_lock{control.gate};
    active = control.active;
    final_sent = control.final_callback_count != 0u;
    quarantined = control.quarantined;
  }
  if (!active) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 2u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (final_sent) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 3u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (quarantined) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 4u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  Coordinate entry{};
  if (!coordinate_at(*owner, signal.identity.coordinate, entry)) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Sequence, 1u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (entry.local_count > std::numeric_limits<std::uint32_t>::digits) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Sequence, 4u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  MetalSequence &sequence = *entry.sequence;
  MetalResidencySlidingGate &gate = sequence.residency_sliding;
  void *const descriptor_destination = [gate.descriptor contents];
  void *const guard = [sequence.guard_zero contents];
  void *const control_destination = [sequence.control contents];
  const std::uint64_t active_mask =
      entry.local_count == std::numeric_limits<std::uint32_t>::digits
          ? std::numeric_limits<std::uint32_t>::max()
          : (std::uint64_t{1u} << entry.local_count) - 1u;
  const BackendResidencySlidingDescriptor descriptor{
      .owner = &sequence,
      .plan_identity = signal.identity.plan_identity,
      .token = signal.identity.token,
      .generation = signal.identity.generation,
      .coordinate = signal.identity.coordinate,
      .turn = signal.identity.turn,
      .read_mask = signal.admission.ok ? active_mask : 0u,
      .write_mask = signal.admission.ok ? active_mask : 0u,
      .descriptor_generation = signal.identity.descriptor_generation,
      .control_generation = signal.identity.control_generation,
      .stride = owner->prepared.width,
      .slot = signal.identity.slot,
  };
  MetalResidencySlidingPayload payload{};
  if (descriptor_destination == nullptr) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Command, 1u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (guard == nullptr) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Command, 2u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (control_destination == nullptr) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Command, 3u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (!metal_residency_sliding::BuildPayload(
          sequence, gate, descriptor,
          std::span<const std::uint32_t>{entry.locals.data(),
                                         entry.local_count},
          payload)) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Encode, 1u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (!persistent_sliding_accept_ready_signal(control, signal)) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::Control,
        signal.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Signal, 1u,
        signal.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  entry.admission = signal.admission;
  *static_cast<std::uint32_t *>(guard) = 1u;
  // A failed-admission coordinate deliberately leaves every payload command
  // guarded.  Consequently the GPU does not advance this slot's control
  // generation.  Seed the exact predecessor for every coordinate so a
  // no-write suffix longer than the role width remains authenticated when the
  // slot is reused.
  const PreparedPipelineControl initial{
      .generation = signal.identity.control_generation -
                    sequence.control_generation_stride,
  };
  std::memcpy(control_destination, &initial, sizeof(initial));
  std::memcpy(descriptor_destination, &payload, sizeof(payload));
  std::atomic_thread_fence(std::memory_order_release);
  metal_residency_sliding::CommitDescriptor(gate, descriptor);
  owner->coordinates[static_cast<std::size_t>(
      signal.identity.coordinate % SlotCount)] = entry;
  owner->ready[signal.identity.slot].signaledValue = entry.ready_value;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
