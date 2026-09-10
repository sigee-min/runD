#include "../internal.hpp"

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

PersistentResidencySlidingSubmitResult
submit_continuation(const PersistentResidencySlidingRequest &request,
                    PersistentResidencySlidingControl &control, Owner &owner,
                    VulkanAdapter &adapter,
                    VulkanResidencyPersistentRun &run) noexcept {
  if (!persistent_sliding_chunk_valid(owner.capability, request) ||
      request.first_coordinate != control.accepted_end ||
      request.chunk_count != control.chunk_span ||
      run.timeline_generation == 0u) {
    return submit_out(invalid());
  }
  const std::uint64_t first = request.first_coordinate;
  const std::uint64_t end = first + request.chunk_count;
  std::array<VulkanTimelineBatch, BatchCount> batches{};
  rund::AccelCheck prepared{true, "ok"};
  for (std::uint64_t coordinate = first; prepared.ok && coordinate < end;
       ++coordinate) {
    VulkanTimelinePoint current{};
    prepared = ReserveVulkanTimelinePoint(adapter.timeline,
                                          run.timeline_generation, current);
    if (!prepared.ok) {
      break;
    }
    VulkanResidencyPersistentRole &native = native_role(run, coordinate);
    const std::size_t cell =
        static_cast<std::size_t>(coordinate % VulkanResidencyWindowCapacity);
    if (coordinate < VulkanResidencyWindowCapacity) {
      run.ready_base[cell] = current.ready_value;
    }
    owner.batches[static_cast<std::size_t>(coordinate % BatchCount)] =
        VulkanTimelineBatch{
            .commands = std::span<const VkCommandBuffer>{native.commands.data(),
                                                         native.command_count},
            .point = current,
        };
    batches[static_cast<std::size_t>(coordinate - first)] =
        owner.batches[static_cast<std::size_t>(coordinate % BatchCount)];
  }
  if (!prepared.ok) {
    quarantine_unknown_locked(control, &owner, &run, &adapter, prepared);
    return submit_out({false, "compute_device_lost"},
                      PersistentResidencySlidingSubmitEvent::Unknown);
  }
  std::uint64_t queue_calls = 0u;
  const rund::AccelCheck submitted = SubmitVulkanTimelineStream(
      adapter,
      std::span<const VulkanTimelineBatch>{
          batches.data(), static_cast<std::size_t>(request.chunk_count)},
      queue_calls, true);
  if (!submitted.ok || queue_calls != 1u) {
    quarantine_unknown_locked(
        control, &owner, &run, &adapter,
        submitted.ok ? rund::AccelCheck{false, "compute_device_lost"}
                     : submitted);
    return submit_out({false, "compute_device_lost"},
                      PersistentResidencySlidingSubmitEvent::Unknown);
  }
  return submit_out({true, "ok"},
                    PersistentResidencySlidingSubmitEvent::Accepted);
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
