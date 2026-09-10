#include <accel/device.hpp>

#include "local.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#include "src/accel/vulkan/command/resources.hpp"
#include "src/accel/vulkan/timeline/owner.hpp"
#include <node/accel/pick.hpp>

#include <array>
#include <chrono>
#include <span>
#include <string_view>
#include <vector>

namespace node_accel_contract {

[[nodiscard]] bool VulkanTimelineFoundationContract() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  namespace detail = rund::node::accel::detail;
  const detail::VulkanTimelineOwner unavailable{};
  const rund::AccelCheck unavailable_check =
      detail::VulkanTimelineCapability(unavailable);
  if (unavailable_check.ok ||
      std::string_view{unavailable_check.reason} !=
          "compute_backend_unsupported" ||
      detail::QueryVulkanTimelineSupport(VK_NULL_HANDLE, VK_API_VERSION_1_2,
                                         {})) {
    return false;
  }
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(vulkan::RequiredPolicy());
  if (!pick.check.ok) {
    return vulkan::FailureReasonIsPrecise(pick);
  }
  auto *const adapter = static_cast<rund::node::accel::detail::VulkanAdapter *>(
      pick.backend.context);
  if (adapter == nullptr) {
    return false;
  }
  const rund::AccelCheck capability =
      detail::VulkanTimelineCapability(adapter->timeline);
  if (!capability.ok) {
    return std::string_view{capability.reason} == "compute_backend_unsupported";
  }

  std::uint32_t generation = 0u;
  detail::VulkanTimelinePoint point{};
  detail::VulkanTimelinePoint following{};
  if (!detail::ReserveVulkanTimelineGeneration(adapter->timeline, generation)
           .ok ||
      !detail::ReserveVulkanTimelinePoint(adapter->timeline, generation, point)
           .ok ||
      !detail::ReserveVulkanTimelinePoint(adapter->timeline, generation,
                                          following)
           .ok ||
      !point || point.generation != generation || point.ready_cell != 0u ||
      following.ready_cell != 1u || point.ready_value != 1u ||
      following.ready_value != 1u || following.value != point.value + 1u ||
      detail::SubmitVulkanTimeline(*adapter, std::span<const VkCommandBuffer>{},
                                   following)
          .ok ||
      detail::SignalVulkanTimelineReady(adapter->timeline, following).ok ||
      !detail::SubmitVulkanTimeline(*adapter,
                                    std::span<const VkCommandBuffer>{}, point)
           .ok ||
      !detail::SignalVulkanTimelineReady(adapter->timeline, point).ok ||
      !detail::WaitVulkanTimelineDone(
           adapter->timeline, point,
           static_cast<std::uint64_t>(std::chrono::seconds{5}.count()) *
               1'000'000'000u)
           .ok ||
      !detail::SubmitVulkanTimeline(
           *adapter, std::span<const VkCommandBuffer>{}, following)
           .ok ||
      !detail::SignalVulkanTimelineReady(adapter->timeline, following).ok ||
      !detail::WaitVulkanTimelineDone(
           adapter->timeline, following,
           static_cast<std::uint64_t>(std::chrono::seconds{5}.count()) *
               1'000'000'000u)
           .ok) {
    return false;
  }
  std::uint64_t ready = 0u;
  std::uint64_t done = 0u;
  if (!detail::ReadVulkanTimeline(adapter->timeline, ready, done).ok ||
      ready < following.ready_value || done < following.value ||
      !detail::CloseVulkanTimelineGeneration(adapter->timeline, following).ok) {
    return false;
  }

  std::uint32_t next_generation = 0u;
  detail::VulkanTimelinePoint next{};
  const bool next_valid =
      detail::ReserveVulkanTimelineGeneration(adapter->timeline,
                                              next_generation)
          .ok &&
      next_generation != generation &&
      detail::ReserveVulkanTimelinePoint(adapter->timeline, next_generation,
                                         next)
          .ok &&
      next.value == following.value + 1u && next.ready_cell == 0u &&
      next.ready_value == point.ready_value + 1u &&
      !detail::SignalVulkanTimelineReady(adapter->timeline, point).ok &&
      detail::SubmitVulkanTimeline(*adapter, std::span<const VkCommandBuffer>{},
                                   next)
          .ok &&
      detail::SignalVulkanTimelineReady(adapter->timeline, next).ok &&
      detail::WaitVulkanTimelineDone(
          adapter->timeline, next,
          static_cast<std::uint64_t>(std::chrono::seconds{5}.count()) *
              1'000'000'000u)
          .ok &&
      detail::CloseVulkanTimelineGeneration(adapter->timeline, next).ok;
  if (!next_valid) {
    return false;
  }

  detail::VulkanCommand empty{};
  if (!detail::CreateCommand(adapter->device, adapter->compute_queue_family,
                             empty, detail::CommandKind::ReusablePrimary)
           .ok ||
      !detail::BeginCommand(adapter->device, empty,
                            detail::CommandKind::ReusablePrimary)
           .ok ||
      !detail::EndCommand(empty).ok) {
    detail::DestroyCommand(adapter->device, empty);
    return false;
  }
  constexpr std::uint64_t timeout =
      static_cast<std::uint64_t>(std::chrono::seconds{5}.count()) *
      1'000'000'000u;
  const std::array<VkCommandBuffer, 1u> commands{empty.buffer};
  bool long_valid = true;
  for (std::size_t sample = 0u; long_valid && sample < 16u; ++sample) {
    std::uint32_t single_generation = 0u;
    detail::VulkanTimelinePoint single{};
    std::uint64_t queue_calls = 0u;
    long_valid = detail::ReserveVulkanTimelineGeneration(adapter->timeline,
                                                         single_generation)
                     .ok &&
                 detail::ReserveVulkanTimelinePoint(adapter->timeline,
                                                    single_generation, single)
                     .ok;
    const detail::VulkanTimelineBatch submitted{
        .commands = commands,
        .point = single,
    };
    long_valid =
        long_valid &&
        detail::SubmitVulkanTimelineWindow(
            *adapter,
            std::span<const detail::VulkanTimelineBatch>{&submitted, 1u},
            queue_calls)
            .ok &&
        queue_calls == 1u &&
        detail::SignalVulkanTimelineReady(adapter->timeline, single).ok &&
        detail::WaitVulkanTimelineDone(adapter->timeline, single, timeout).ok &&
        detail::CloseVulkanTimelineGeneration(adapter->timeline, single).ok;
  }
  std::uint32_t window_generation = 0u;
  std::array<detail::VulkanTimelinePoint, 4u> points{};
  std::array<detail::VulkanTimelineBatch, 4u> batches{};
  long_valid = long_valid && detail::ReserveVulkanTimelineGeneration(
                                 adapter->timeline, window_generation)
                                 .ok;
  for (std::size_t index = 0u; long_valid && index < points.size(); ++index) {
    long_valid = detail::ReserveVulkanTimelinePoint(
                     adapter->timeline, window_generation, points[index])
                     .ok;
    batches[index] = detail::VulkanTimelineBatch{.commands = commands,
                                                 .point = points[index]};
  }
  std::uint64_t queue_calls = 0u;
  long_valid =
      long_valid && points[0u].ready_value > points[1u].ready_value &&
      points[1u].ready_value >= 1u && points[2u].ready_value >= 1u &&
      points[3u].ready_value >= 1u &&
      detail::SubmitVulkanTimelineWindow(*adapter, batches, queue_calls).ok &&
      queue_calls == 1u &&
      detail::SignalVulkanTimelineReady(adapter->timeline, points[1u]).ok &&
      detail::SignalVulkanTimelineReady(adapter->timeline, points[0u]).ok &&
      detail::SignalVulkanTimelineReady(adapter->timeline, points[2u]).ok &&
      detail::SignalVulkanTimelineReady(adapter->timeline, points[3u]).ok &&
      detail::WaitVulkanTimelineDone(adapter->timeline, points.back(), timeout)
          .ok &&
      detail::CloseVulkanTimelineGeneration(adapter->timeline, points.back())
          .ok;
  constexpr std::size_t stream_count = 17u;
  std::uint32_t stream_generation = 0u;
  std::vector<detail::VulkanTimelinePoint> stream_points{};
  std::vector<detail::VulkanTimelineBatch> stream_batches{};
  try {
    stream_points.resize(stream_count);
    stream_batches.resize(stream_count);
  } catch (...) {
    detail::DestroyCommand(adapter->device, empty);
    return false;
  }
  long_valid = long_valid && detail::ReserveVulkanTimelineGeneration(
                                 adapter->timeline, stream_generation)
                                 .ok;
  for (std::size_t index = 0u; long_valid && index < stream_count; ++index) {
    long_valid = detail::ReserveVulkanTimelinePoint(
                     adapter->timeline, stream_generation, stream_points[index])
                     .ok &&
                 stream_points[index].ready_cell == index % 4u &&
                 (index == 0u || stream_points[index].value ==
                                     stream_points[index - 1u].value + 1u);
    stream_batches[index] = detail::VulkanTimelineBatch{
        .commands = commands, .point = stream_points[index]};
  }
  queue_calls = 9u;
  long_valid =
      long_valid &&
      !detail::SubmitVulkanTimelineWindow(*adapter, stream_batches, queue_calls)
           .ok &&
      queue_calls == 0u &&
      detail::SubmitVulkanTimelineStream(*adapter, stream_batches, queue_calls)
          .ok &&
      queue_calls == 1u &&
      // Reusing a ready cell cannot skip its earlier unsignalled generation.
      !detail::SignalVulkanTimelineReady(adapter->timeline, stream_points[4u])
           .ok &&
      // Independent cells may become ready out of global epoch order.
      detail::SignalVulkanTimelineReady(adapter->timeline, stream_points[1u])
          .ok &&
      detail::SignalVulkanTimelineReady(adapter->timeline, stream_points[0u])
          .ok;
  for (std::size_t index = 2u; long_valid && index < stream_count; ++index) {
    long_valid = detail::SignalVulkanTimelineReady(adapter->timeline,
                                                   stream_points[index])
                     .ok;
  }
  long_valid = long_valid &&
               detail::WaitVulkanTimelineDone(adapter->timeline,
                                              stream_points.back(), timeout)
                   .ok &&
               detail::CloseVulkanTimelineGeneration(adapter->timeline,
                                                     stream_points.back())
                   .ok;
  detail::DestroyCommand(adapter->device, empty);
  if (!long_valid) {
    return false;
  }

  // An idle completion worker must not retain the only public owner across
  // its condition-variable wait. Releasing a transient pick synchronously
  // runs the neutral-thread-safe stop/join path and expires the owner.
  rund::AccelDevice transient =
      rund::node::accel::PickAccel(vulkan::RequiredPolicy());
  if (!transient.check.ok) {
    return vulkan::FailureReasonIsPrecise(transient);
  }
  const std::weak_ptr<void> transient_owner = transient.owner;
  transient = {};
  return transient_owner.expired();
#else
  return true;
#endif
}

} // namespace node_accel_contract
