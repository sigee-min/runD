#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../sliding/internal.hpp"

#include <cstring>
#include <limits>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool acquired_result(
    const Coordinate &entry,
    MetalPersistentResidencySlidingDiagnosticTrace &trace) noexcept {
  trace.coordinate = entry.identity.coordinate;
  trace.admission = entry.admission.ok;
  trace.expected_descriptor_generation = entry.identity.descriptor_generation;
  trace.expected_control_generation = entry.identity.control_generation;
  if (entry.sequence == nullptr) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate = MetalPersistentResidencySlidingDiagnosticPredicate::State;
    trace.predicate_key = 1u;
    return false;
  }
  MetalResidencySlidingGate &gate = entry.sequence->residency_sliding;
  if (gate.descriptor == nil ||
      gate.descriptor.length < sizeof(MetalResidencySlidingPayload)) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::Payload;
    trace.predicate_key = 1u;
    return false;
  }
  const auto *const payload = static_cast<const MetalResidencySlidingPayload *>(
      [gate.descriptor contents]);
  if (payload == nullptr) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::Payload;
    trace.predicate_key = 2u;
    return false;
  }
  gate.gpu_result_read_count.fetch_add(1u, std::memory_order_relaxed);
  const std::uint64_t expected_generation =
      static_cast<std::uint64_t>(
          payload->words[metal_residency_sliding::DescriptorGenerationWord]) |
      (static_cast<std::uint64_t>(
           payload
               ->words[metal_residency_sliding::DescriptorGenerationWord + 1u])
       << 32u);
  const std::uint64_t observed_generation =
      static_cast<std::uint64_t>(
          payload->words[MetalResidencySlidingObservedGenerationWord]) |
      (static_cast<std::uint64_t>(
           payload->words[MetalResidencySlidingObservedGenerationWord + 1u])
       << 32u);
  trace.observed_descriptor_generation = observed_generation;
  trace.accepted = payload->words[MetalResidencySlidingAcceptedWord] == 1u;
  trace.reason = payload->words[MetalResidencySlidingReasonWord];
  if (expected_generation != entry.identity.descriptor_generation ||
      observed_generation != expected_generation) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate = MetalPersistentResidencySlidingDiagnosticPredicate::
        DescriptorGeneration;
    trace.predicate_key = observed_generation;
    return false;
  }
  if (payload->words[MetalResidencySlidingAcceptedWord] != 1u ||
      payload->words[MetalResidencySlidingReasonWord] != 0u) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::AcceptedReason;
    trace.predicate_key = trace.reason;
    return false;
  }
  if (!entry.admission.ok) {
    return true;
  }
  PreparedPipelineControl control{};
  void *const source = [entry.sequence->control contents];
  if (source == nullptr) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::Control;
    trace.predicate_key = 1u;
    return false;
  }
  std::memcpy(&control, source, sizeof(control));
  trace.observed_control_generation = control.generation;
  if (control.generation != entry.identity.control_generation) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::ControlGeneration;
    trace.predicate_key = control.generation;
    return false;
  }
  return true;
}

} // namespace

rund::AccelCheck
wait_done(PersistentResidencySlidingControl &control,
          const PersistentResidencySlidingDoneWait &wait,
          PersistentResidencySlidingDoneObservation &observation) noexcept {
  observation = {};
  const Binding binding = bind(control);
  const std::shared_ptr<Owner> owner = binding.owner;
  if (owner == nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  Coordinate entry{};
  {
    std::lock_guard lock{owner->gate};
    if (!valid_binding(binding, *owner, control) ||
        wait.identity.coordinate >= owner->prepared.coordinate_count ||
        !coordinate_at(*owner, wait.identity.coordinate, entry)) {
      record_first_failure(
          *owner, MetalPersistentResidencySlidingDiagnosticStage::WaitPayload,
          wait.identity.coordinate,
          MetalPersistentResidencySlidingDiagnosticPredicate::Identity, 1u,
          wait.identity.coordinate);
      return {false, "accel_kernel_pipeline_invalid"};
    }
    const Coordinate &cached = owner->coordinates[static_cast<std::size_t>(
        wait.identity.coordinate % SlotCount)];
    if (persistent_sliding_same_service_identity(cached.identity,
                                                 entry.identity)) {
      entry.admission = cached.admission;
    }
    bool closed = false;
    {
      std::lock_guard control_lock{control.gate};
      closed = !control.active || control.final_callback_count != 0u ||
               control.quarantined;
    }
    if (closed ||
        !persistent_sliding_same_service_identity(entry.identity,
                                                  wait.identity) ||
        !persistent_sliding_accept_done_wait(control, wait)) {
      record_first_failure(
          *owner, MetalPersistentResidencySlidingDiagnosticStage::WaitPayload,
          wait.identity.coordinate,
          MetalPersistentResidencySlidingDiagnosticPredicate::Identity, 1u,
          wait.identity.coordinate);
      return {false, "accel_kernel_pipeline_invalid"};
    }
  }
  const bool signaled =
      [owner->done waitUntilSignaledValue:entry.done_value
                                timeoutMS:DoneWaitMilliseconds];
  std::lock_guard lock{owner->gate};
  if (!valid_binding(binding, *owner, control)) {
    record_first_failure(
        *owner, MetalPersistentResidencySlidingDiagnosticStage::WaitPayload,
        wait.identity.coordinate,
        MetalPersistentResidencySlidingDiagnosticPredicate::Identity, 3u,
        wait.identity.coordinate);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  bool retired = true;
  std::uint64_t accepted_end = owner->prepared.coordinate_count;
  {
    std::lock_guard control_lock{control.gate};
    accepted_end = persistent_sliding_accepted_end(control);
  }
  if (signaled && wait.identity.coordinate + 1u == accepted_end &&
      !owner->native_retired) {
    owner->native_retired = EndMetalResidencyCommand(*owner->adapter);
    retired = owner->native_retired;
    // The single native commit is counted at the commit site.  Retirement
    // contributes only its wait duration; counting again here would report
    // two submits for one MTLCommandBuffer commit.
    RecordMetalCommandSubmitWaitOnlyNs(
        *owner->adapter, MonotonicNanoseconds() - owner->submit_begin_ns);
  }
  MetalPersistentResidencySlidingDiagnosticTrace trace{};
  trace.coordinate = entry.identity.coordinate;
  trace.signaled = signaled;
  trace.retired = retired;
  trace.admission = entry.admission.ok;
  trace.expected_descriptor_generation = entry.identity.descriptor_generation;
  trace.expected_control_generation = entry.identity.control_generation;
  const bool acquired =
      signaled && retired ? acquired_result(entry, trace) : false;
  if (!signaled) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitSignal;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::Signal;
    trace.predicate_key = 1u;
  } else if (!retired) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitRetire;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::Retire;
    trace.predicate_key = 1u;
  }
  record_wait_trace(*owner, trace);
  if (!acquired) {
    record_first_failure(*owner, trace);
  }
  const bool known = signaled && retired && acquired;
  observation = PersistentResidencySlidingDoneObservation{
      .identity = wait.identity,
      .check = known ? entry.admission
                     : rund::AccelCheck{false, "compute_device_lost"},
      .terminal =
          known ? NativeTerminal::Known : NativeTerminal::UnknownMayWrite,
      .dispatched = entry.admission.ok,
      .completed = known,
      .may_write = !known || entry.admission.ok,
  };
  if (!persistent_sliding_accept_done_observation(control, wait, observation)) {
    trace.stage = MetalPersistentResidencySlidingDiagnosticStage::WaitPayload;
    trace.predicate =
        MetalPersistentResidencySlidingDiagnosticPredicate::Identity;
    trace.predicate_key = 2u;
    record_wait_trace(*owner, trace);
    record_first_failure(*owner, trace);
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (known) {
    std::lock_guard control_lock{control.gate};
    ++control.gpu_completed_coordinates;
    if (entry.admission.ok && control.first_failure.ok) {
      ++control.completed_prefix;
    } else if (!entry.admission.ok) {
      if (control.first_failure.ok) {
        control.first_failure = entry.admission;
        control.first_failure_coordinate = entry.identity.coordinate;
      }
      if (control.suppressed_count == 0u) {
        control.suppressed_first = entry.identity.coordinate;
      }
      ++control.suppressed_count;
    }
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
