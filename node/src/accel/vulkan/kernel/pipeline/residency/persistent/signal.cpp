#include "../../../../buffer/access.hpp"

#include "../mode.hpp"
#include "../generated_indirect/map.hpp"
#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck SignalVulkanResidencyPersistentReady(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingReadySignal &signal) noexcept {
  using namespace vulkan_persistent_detail;
  std::unique_lock control_lock{control.gate};
  VulkanResidencyPersistentRun *const run = active_run(control);
  if (run == nullptr || !control.active || control.quarantined ||
      signal.identity.plan_identity != control.plan_identity ||
      signal.identity.token != control.token ||
      signal.identity.generation != control.generation ||
      signal.identity.coordinate != control.next_ready_coordinate ||
      signal.identity.coordinate >= persistent_sliding_accepted_end(control) ||
      (!signal.admission.ok && (signal.admission.reason == nullptr ||
                                signal.admission.reason[0] == '\0')) ||
      (control.failed_admission_count != 0u && signal.admission.ok) ||
      (control.service_failed && signal.admission.ok) ||
      (signal.identity.coordinate >= control.width &&
       signal.identity.coordinate - control.width >=
           control.next_acknowledgement_coordinate)) {
    return invalid();
  }
  PersistentResidencySlidingServiceIdentity expected{};
  if (!persistent_sliding_service_identity(control, signal.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, signal.identity)) {
    return invalid();
  }
  std::lock_guard run_lock{run->gate};
  if (!run->active || run->final_sent || run->quarantined) {
    return invalid();
  }
  const std::uint64_t coordinate = signal.identity.coordinate;
  const std::size_t cell_index =
      static_cast<std::size_t>(coordinate % VulkanResidencyWindowCapacity);
  VulkanResidencyPersistentCell &cell = run->cells[cell_index];
  VulkanResidencyPersistentRole &native = native_role(*run, coordinate);
  const PersistentResidencySlidingRole &source = source_role(*run, coordinate);
  VulkanPipeline *const pipeline = native.pipeline;
  const VulkanTimelinePoint ready = point(*run, coordinate);
  // A recurrent stream deliberately reuses the four bounded authentication
  // cells.  A live prior occupant is forbidden, but an observed and
  // acknowledged occupant is the exact release frontier that makes reuse
  // safe.
  if (pipeline == nullptr || (cell.signaled && !cell.acknowledged) ||
      pipeline->residency == nullptr ||
      !CanSignalVulkanTimelineReady(pipeline->adapter->timeline, ready).ok) {
    return invalid();
  }
  const auto access =
      vulkan_generated_indirect_detail::map_access(*pipeline->residency);
  if (!access.usable()) {
    return invalid();
  }
  const std::size_t count = local_count(*run, coordinate);
  const std::uint64_t active =
      count == std::numeric_limits<std::uint64_t>::digits
          ? std::numeric_limits<std::uint64_t>::max()
          : (std::uint64_t{1u} << count) - 1u;
  const BackendResidencySlidingDescriptor descriptor{
      .owner = pipeline,
      .plan_identity = signal.identity.plan_identity,
      .token = signal.identity.token,
      .generation = signal.identity.generation,
      .coordinate = coordinate,
      .turn = signal.identity.turn,
      .read_mask = active,
      .write_mask = active,
      .descriptor_generation = signal.identity.descriptor_generation,
      .control_generation = signal.identity.control_generation,
      .stride = run->request.width,
      .slot = signal.identity.slot,
  };
  VulkanResidencySlidingGate &gate = pipeline->residency->sliding;
  const bool generated = access.selected;
  VulkanResidencySlidingPayload payload{};
  std::lock_guard adapter_lock{pipeline->adapter->mutex};
  if (pipeline->adapter->residency_quarantined.load(
          std::memory_order_acquire) ||
      !vulkan_sliding_detail::build_payload(
          *pipeline, gate, descriptor,
          std::span<const std::uint32_t>{source.locals.data(), count},
          payload) ||
      !UploadVulkanBuffer(gate.descriptor, &payload, sizeof(payload)) ||
      (!generated &&
       !SelectVulkanResidencyArguments(
           *pipeline,
           std::span<const std::uint32_t>{source.locals.data(), count},
           signal.admission.ok)) ||
      !SeedVulkanResidencyControl(
          *pipeline,
          BackendResidencyWindowSignal{
              .admission = signal.admission,
              .plan_identity = signal.identity.plan_identity,
              .token = signal.identity.token,
              .generation = signal.identity.generation,
              .epoch = coordinate,
              .control_generation = signal.identity.control_generation,
              .bank = static_cast<std::uint8_t>(coordinate % 2u),
          },
          signal.admission.ok)) {
    return unavailable();
  }
  std::atomic_thread_fence(std::memory_order_release);
  const rund::AccelCheck signaled =
      SignalVulkanTimelineReady(pipeline->adapter->timeline, ready);
  if (!signaled.ok) {
    return signaled;
  }
  vulkan_sliding_detail::commit_descriptor(gate, descriptor);
  cell = VulkanResidencyPersistentCell{
      .admission = signal.admission,
      .coordinate = coordinate,
      .signaled = true,
  };
  if (!signal.admission.ok) {
    if (control.failed_admission_count == 0u) {
      control.first_failed_admission_coordinate = coordinate;
    }
    ++control.failed_admission_count;
  }
  ++control.next_ready_coordinate;
  ++control.backing_signal_count;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
