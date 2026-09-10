#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

[[nodiscard]] bool DeviceVsmQueueCount(const rund::AccelDevice &pick,
                                       std::uint64_t &count) noexcept {
  const std::shared_ptr<accel::PickToken> token = accel::AdmitPick(pick);
  accel::VulkanAdapter *const adapter =
      token == nullptr ? nullptr : accel::CheckedVulkanAdapter(token->raw);
  if (adapter == nullptr) {
    return false;
  }
  std::lock_guard lock{adapter->mutex};
  count = adapter->command_submit_count;
  return true;
}

PersistentBacking::PersistentBacking(const std::size_t bytes)
    : bytes_(bytes), write_count_{} {}

std::uint64_t PersistentBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::Status
PersistentBacking::read(const std::uint64_t offset,
                        const std::span<std::byte> output) noexcept {
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  return rund::compute::Status::success();
}

rund::compute::Status
PersistentBacking::write(const std::uint64_t offset,
                         const std::span<const std::byte> input) noexcept {
  if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  write_count_.fetch_add(1u, std::memory_order_relaxed);
  return rund::compute::Status::success();
}

std::uint64_t PersistentBacking::write_count() const noexcept {
  return write_count_.load(std::memory_order_relaxed);
}

void CompletePersistent(
    void *const raw, accel::PersistentResidencySlidingFinal &&final) noexcept {
  auto *const wait = static_cast<FinalWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->final = std::move(final);
  wait->callback_count.fetch_add(1u, std::memory_order_release);
}

[[nodiscard]] std::shared_ptr<void> NativeBackend(
    const std::shared_ptr<rund::compute::detail::PipelineState> &pipeline) {
  auto *const common = pipeline == nullptr || !pipeline->prepared.ok
                           ? nullptr
                           : static_cast<accel::prepared::PipelineState *>(
                                 pipeline->prepared.owner.get());
  return common == nullptr ? std::shared_ptr<void>{} : common->backend;
}

[[nodiscard]] bool ProductQueueCount(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    std::uint64_t &count) noexcept {
  count = 0u;
  const std::shared_ptr<void> backend = state == nullptr
                                            ? std::shared_ptr<void>{}
                                            : NativeBackend(state->pipeline);
  auto *const native = static_cast<accel::VulkanPipeline *>(backend.get());
  if (native == nullptr || native->adapter == nullptr) {
    return false;
  }
  std::lock_guard lock{native->adapter->mutex};
  count = native->adapter->command_submit_count;
  return true;
}

[[nodiscard]] bool
PersistentRawControlCleared(const std::shared_ptr<void> &prepared) noexcept {
  auto *const pipeline = static_cast<accel::VulkanPipeline *>(prepared.get());
  if (pipeline == nullptr || pipeline->residency == nullptr) {
    return false;
  }
  std::lock_guard lock{pipeline->residency->persistent.gate};
  return pipeline->residency->persistent.control == nullptr;
}

[[nodiscard]] bool PersistentControlPristine(
    const accel::PersistentResidencySlidingControl &control) noexcept {
  return control.native == nullptr && control.plan_identity == 0u &&
         control.token == 0u && control.generation == 0u &&
         control.coordinate_count == 0u &&
         control.next_ready_coordinate == 0u &&
         control.next_wait_coordinate == 0u &&
         control.next_observation_coordinate == 0u &&
         control.next_acknowledgement_coordinate == 0u &&
         control.native_submit_count == 0u &&
         control.epoch_native_submit_count == 0u &&
         control.backend_epoch_callback_count == 0u &&
         control.backing_wait_count == 0u &&
         control.backing_signal_count == 0u &&
         control.backing_acknowledgement_count == 0u &&
         control.backing_service_failure_count == 0u &&
         control.failed_admission_count == 0u &&
         control.first_failed_admission_coordinate ==
             accel::PersistentSlidingNoCoordinate &&
         control.first_service_failure_coordinate ==
             accel::PersistentSlidingNoCoordinate &&
         control.width == 0u && !control.active && !control.service_failed &&
         !control.quarantined;
}

[[nodiscard]] bool
CommitAccepted(const accel::PersistentResidencySlidingCapability &capability,
               const accel::PersistentResidencySlidingRequest &request,
               accel::PersistentResidencySlidingControl &control) noexcept {
  const bool whole =
      request.mode == accel::PersistentResidencySlidingMode::OneSubmit;
  const std::uint64_t first = whole ? 0u : request.first_coordinate;
  const std::uint64_t count =
      whole ? request.coordinate_count : request.chunk_count;
  std::lock_guard lock{control.gate};
  return accel::persistent_sliding_commit_accept_locked(capability, request,
                                                        control, first, count);
}

[[nodiscard]] bool RejectOversizedCapability(
    accel::VulkanAdapter &adapter,
    const accel::PersistentResidencySlidingRequest &request) noexcept {
  std::lock_guard lock{adapter.mutex};
  const std::uint64_t original = adapter.persistent_stream_submit_capacity;
  adapter.persistent_stream_submit_capacity =
      std::numeric_limits<std::uint64_t>::max();
  const accel::PersistentResidencySlidingCapability rejected =
      accel::VulkanResidencyPersistentCapability(
          std::span<const accel::PersistentResidencySlidingRole>{
              request.roles.data(), request.width},
          static_cast<std::uint64_t>(
              std::numeric_limits<std::uint32_t>::max()) +
              1u,
          request.memory);
  adapter.persistent_stream_submit_capacity = original;
  return !rejected.check.ok && rejected.check.reason != nullptr &&
         std::strcmp(rejected.check.reason, "compute_pipeline_capacity") == 0;
}

[[nodiscard]] bool SameFusedDirectStorage(
    const accel::VulkanFusedDirectRecurrenceDiagnostics &left,
    const accel::VulkanFusedDirectRecurrenceDiagnostics &right) noexcept {
  return left.native_command_buffer_count ==
             right.native_command_buffer_count &&
         left.native_dispatch_count == right.native_dispatch_count &&
         left.descriptor_set_count == right.descriptor_set_count &&
         left.device_buffer_bytes == right.device_buffer_bytes &&
         left.route_host_bytes == right.route_host_bytes &&
         left.push_constant_bytes == right.push_constant_bytes &&
         !left.command_buffer_allocation_bytes_observable &&
         !right.command_buffer_allocation_bytes_observable;
}

[[nodiscard]] bool
SameCurrentCommonMemory(const accel::PreparedPipelineMemory &left,
                        const accel::PreparedPipelineMemory &right) noexcept {
  return left.host.current == right.host.current &&
         left.device.current == right.device.current &&
         left.staging.current == right.staging.current;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
