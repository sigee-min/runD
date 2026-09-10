#include "../../buffer/access.hpp"
#include "../../buffer/create.hpp"

#include "internal.hpp"

#include "../copy.hpp"

#include "../../../kernel/recurrence.hpp"
#include "../../buffer/resident/find.hpp"
#include "../../resident/access.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanWindowResources(
    VulkanAdapter &adapter, const std::span<const BackendBatchEntry> entries,
    const std::uint64_t dispatch_capacity, const std::uint64_t gate_capacity,
    const PreparedPipelineStatusLayout &status, const VulkanBuffer &control,
    VulkanWindowResources &resources) {
  std::size_t route_count = 0u;
  std::uint32_t state_count = 0u;
  for (const BackendBatchEntry &entry : entries) {
    const BackendWindow *const window = entry.recurrence.window;
    if (window == nullptr) {
      continue;
    }
    if (window->state == std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    ++route_count;
    state_count = std::max(state_count, window->state + 1u);
  }
  if (route_count == 0u) {
    return gate_capacity == 0u
               ? rund::AccelCheck{true, "ok"}
               : rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  if (dispatch_capacity == 0u || control.buffer == VK_NULL_HANDLE ||
      control.bytes < sizeof(PreparedPipelineControl) ||
      gate_capacity > dispatch_capacity ||
      dispatch_capacity > std::numeric_limits<std::uint32_t>::max() ||
      dispatch_capacity > std::numeric_limits<std::uint32_t>::max() / 3u ||
      dispatch_capacity > std::numeric_limits<std::size_t>::max() /
                              sizeof(VkDispatchIndirectCommand) ||
      dispatch_capacity >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.adapter = &adapter;
  resources.gate_capacity = gate_capacity;
  resources.state_count = state_count;
  const std::uint64_t state_bytes =
      static_cast<std::uint64_t>(state_count) * sizeof(ResidentState);
  const std::uint64_t argument_bytes =
      dispatch_capacity * sizeof(VkDispatchIndirectCommand);
  const std::uint64_t owner_bytes = dispatch_capacity * sizeof(std::uint32_t);
  if (state_count == 0u ||
      !CreateVulkanBuffer(adapter, state_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          resources.states) ||
      !CreateVulkanBuffer(adapter, argument_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          resources.arguments) ||
      !CreateVulkanBuffer(adapter, argument_bytes,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          resources.original_arguments) ||
      !CreateVulkanBuffer(adapter, owner_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          resources.owners) ||
      !ClearVulkanBuffer(resources.states, state_bytes) ||
      !ClearVulkanBuffer(resources.arguments, argument_bytes) ||
      resources.arguments.mapped == nullptr ||
      resources.owners.mapped == nullptr) {
    DestroyVulkanWindow(resources);
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  {
    resources.original.resize(static_cast<std::size_t>(dispatch_capacity));
    resources.routes.reserve(route_count);
    resources.gates.reserve(static_cast<std::size_t>(gate_capacity));
  }
  std::fill_n(static_cast<std::uint32_t *>(resources.owners.mapped),
              static_cast<std::size_t>(dispatch_capacity),
              std::numeric_limits<std::uint32_t>::max());
  resources.capture.arguments = resources.arguments.buffer;
  resources.capture.mapped =
      static_cast<VkDispatchIndirectCommand *>(resources.arguments.mapped);
  resources.capture.original = resources.original.data();
  resources.capture.owners =
      static_cast<std::uint32_t *>(resources.owners.mapped);
  resources.capture.capacity = resources.original.size();
  resources.capture.context = &resources;
  resources.capture.indirect = &EncodeVulkanWindowIndirect;

  VulkanResidentState &resident = VulkanResidents(adapter);
  {
    std::lock_guard lock{resident.mutex};
    for (std::size_t entry_index = 0u; entry_index < entries.size();
         ++entry_index) {
      const BackendWindow *const window =
          entries[entry_index].recurrence.window;
      if (window == nullptr) {
        continue;
      }
      const bool occurrence_valid =
          window->valid_occurrence(entries[entry_index].transduced_action());
      const std::uint32_t template_index = entries[entry_index].template_index;
      VulkanResidentBufferResult count = ResolveVulkanResidentBuffer(
          resident, window->count.source, window->count.handle,
          "compute_resident_id_invalid", true);
      VulkanCopyRange count_range{};
      const bool window_valid =
          count.check.ok && count.device_buffer != nullptr &&
          occurrence_valid && template_index < status.active_step_count &&
          window->count.source.count == 1u &&
          window->count.source.element_bytes == sizeof(std::uint32_t) &&
          PlanVulkanCopyRange(adapter, window->count.source,
                              count.device_buffer, count_range);
      if (!window_valid) {
        const rund::AccelCheck failed =
            count.check.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                           : count.check;
        DestroyVulkanWindow(resources);
        return failed;
      }
      count.ref = window->count.source;

      std::array<VulkanResidentBufferResult, 3u> terminals{count, count, count};
      std::array<VulkanCopyRange, 3u> terminal_ranges{count_range, count_range,
                                                      count_range};
      if (window->has_terminal) {
        for (std::size_t bank = 0u; bank < terminals.size(); ++bank) {
          const BackendRead &terminal = window->terminal[bank];
          terminals[bank] = ResolveVulkanResidentBuffer(
              resident, terminal.source, terminal.handle,
              "compute_resident_id_invalid", true);
          terminal_ranges[bank] = {};
          if (!terminals[bank].check.ok ||
              terminals[bank].device_buffer == nullptr ||
              terminal.source.count != 1u ||
              terminal.source.element_bytes != sizeof(std::uint32_t) ||
              !PlanVulkanCopyRange(adapter, terminal.source,
                                   terminals[bank].device_buffer,
                                   terminal_ranges[bank])) {
            const rund::AccelCheck failed =
                terminals[bank].check.ok
                    ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                    : terminals[bank].check;
            DestroyVulkanWindow(resources);
            return failed;
          }
          terminals[bank].ref = terminal.source;
        }
      }
      std::uint32_t phase = 0u;
      if (!EncodeBackendWindowPhase(window->phase, phase)) {
        DestroyVulkanWindow(resources);
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      resources.routes.push_back(VulkanWindowRoute{
          .count = count,
          .terminals = terminals,
          .count_binding =
              VulkanStorageBinding{count.device_buffer, count_range.base,
                                   count_range.bytes},
          .terminal_bindings = {VulkanStorageBinding{terminals[0].device_buffer,
                                                     terminal_ranges[0].base,
                                                     terminal_ranges[0].bytes},
                                VulkanStorageBinding{terminals[1].device_buffer,
                                                     terminal_ranges[1].base,
                                                     terminal_ranges[1].bytes},
                                VulkanStorageBinding{terminals[2].device_buffer,
                                                     terminal_ranges[2].base,
                                                     terminal_ranges[2].bytes}},
          .params =
              VulkanWindowParams{
                  .count_offset_words = count_range.offset_words,
                  .terminal_offset_words = {terminal_ranges[0].offset_words,
                                            terminal_ranges[1].offset_words,
                                            terminal_ranges[2].offset_words},
                  .maximum = window->maximum,
                  .tile = window->tile,
                  .iteration = window->outer_iteration,
                  .expected = window->expected,
                  .state = window->state,
                  .has_terminal =
                      static_cast<std::uint32_t>(window->has_terminal),
                  .command_count =
                      static_cast<std::uint32_t>(dispatch_capacity),
                  .phase = phase,
                  .declared_step = status.declared_steps[template_index],
                  .overflow_reason = static_cast<std::uint32_t>(
                      rund::compute::Reason::BoundedCountInvalid),
                  .inner_bound = window->inner_bound,
                  .inner_advance = window->inner_advance,
              },
          .entry = static_cast<std::uint32_t>(entry_index),
      });
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
