#include "window.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "copy.hpp"
#include "lease.hpp"

#include "../../kernel/footprint.hpp"
#include "../buffer/resident/find.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../resident/access.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const std::string &WindowSource() {
  static const std::string source = R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Terminal0 {
  uint terminal0[];
};
layout(set = 0, binding = 1, std430) readonly buffer Terminal1 {
  uint terminal1[];
};
layout(set = 0, binding = 2, std430) readonly buffer Terminal2 {
  uint terminal2[];
};
layout(set = 0, binding = 3, std430) readonly buffer Count {
  uint counts[];
};
layout(set = 0, binding = 4, std430) buffer States {
  uvec2 states[];
};
layout(set = 0, binding = 5, std430) buffer Arguments {
  uint argument_words[];
};
layout(set = 0, binding = 6, std430) readonly buffer Owners {
  uint owners[];
};
layout(set = 0, binding = 7, std430) buffer Control {
  uint control[];
};
layout(push_constant) uniform WindowParams {
  uint64_t count_offset_words;
  uint64_t terminal_offset_words[3];
  uint maximum;
  uint tile;
  uint iteration;
  uint expected;
  uint state;
  uint has_terminal;
  uint command_count;
  uint phase;
  uint declared_step;
  uint overflow_reason;
  uint inner_bound;
} p;
shared uint enabled;
shared uint fresh;
uint64_t load64(uint word) {
  return uint64_t(control[word]) | (uint64_t(control[word + 1u]) << 32u);
}
void store64(uint word, uint64_t value) {
  control[word] = uint(value);
  control[word + 1u] = uint(value >> 32u);
}
void add64(uint word, uint64_t value) {
  const uint64_t current = load64(word);
  const uint64_t maximum = uint64_t(0xfffffffffffffffful);
  store64(word, value > maximum - current ? maximum : current + value);
}
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint phase = p.phase & 3u;
  const bool preflight = (p.phase & 0x80000000u) != 0u;
  if (lane == 0u) {
    const uvec2 current = states[p.state];
    const bool failed = control[1] != 0u;
    if (phase == 2u) {
      enabled = current.y == 0u && !failed ? 1u : 0u;
      fresh = current.y == 0u && failed ? 1u : 0u;
      if (enabled != 0u) { add64(28u, 1ul); }
    } else if (phase == 3u) {
      enabled = current.y == 0u && !failed ? 1u : 0u;
      fresh = current.y == 0u && failed ? 1u : 0u;
      if (enabled != 0u) {
        add64(12u, 1ul);
        add64(24u, 1ul);
      }
    } else if (phase == 1u && !preflight) {
      enabled = current.y == 0u && !failed ? 1u : 0u;
      fresh = current.y == 0u && failed ? 1u : 0u;
    } else {
      const uint items = counts[uint(p.count_offset_words)];
      const uint64_t base = uint64_t(p.iteration) * uint64_t(p.tile);
      uint terminal = terminal0[uint(p.terminal_offset_words[0])];
      if (current.x == 1u) {
        terminal = terminal1[uint(p.terminal_offset_words[1])];
      } else if (current.x == 2u) {
        terminal = terminal2[uint(p.terminal_offset_words[2])];
      }
      const bool ended = p.has_terminal != 0u && terminal == p.expected;
      const bool overflow = current.y == 0u && items > p.maximum;
      if (overflow && !failed) {
        control[1] = p.overflow_reason;
        control[2] = p.declared_step;
        store64(18u, uint64_t(p.maximum));
        control[20] = phase == 1u ? p.iteration : 0xffffffffu;
        control[21] = 0xffffffffu;
        control[22] = phase == 1u ? 1u : 0u;
      }
      enabled = current.y == 0u && control[1] == 0u &&
                        base < uint64_t(items) && !ended
                    ? 1u
                    : 0u;
      fresh = current.y == 0u && enabled == 0u ? 1u : 0u;
      if (phase == 1u && control[1] == 0u && enabled == 0u) {
        add64(14u, 1ul);
        add64(26u, 1ul);
        add64(30u, uint64_t(p.inner_bound));
      }
    }
  }
  barrier();
  if (fresh != 0u) {
    for (uint index = lane; index < p.command_count; index += 256u) {
      if (owners[index] == p.state) {
        const uint word = index * 3u;
        argument_words[word] = 0u;
        argument_words[word + 1u] = 0u;
        argument_words[word + 2u] = 0u;
      }
    }
  }
  barrier();
  if (lane == 0u) {
    if (enabled != 0u) {
      if (phase == 0u || phase == 3u) {
        states[p.state].x = 1u + (p.iteration & 1u);
      }
    } else if (states[p.state].y == 0u) {
      states[p.state].y = p.iteration + 1u;
    }
  }
}
)GLSL";
  return source;
}

struct GateParams final {
  std::uint32_t state{};
  std::uint32_t source_word{};
  std::uint32_t target_word{};
};

static_assert(sizeof(GateParams) == 12u);

[[nodiscard]] const std::string &GateSource() {
  static const std::string source = R"GLSL(#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Source {
  uint source_words[];
};
layout(set = 0, binding = 1, std430) buffer Target {
  uint target_words[];
};
layout(set = 0, binding = 2, std430) readonly buffer States {
  uvec2 states[];
};
layout(push_constant) uniform GateParams {
  uint state;
  uint source_word;
  uint target_word;
} p;
void main() {
  const bool enabled = states[p.state].y == 0u;
  target_words[p.target_word] = enabled ? source_words[p.source_word] : 0u;
  target_words[p.target_word + 1u] =
      enabled ? source_words[p.source_word + 1u] : 0u;
  target_words[p.target_word + 2u] =
      enabled ? source_words[p.source_word + 2u] : 0u;
}
)GLSL";
  return source;
}

[[nodiscard]] VulkanCollectivePipeline *
AcquireWindowPipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x7265736964656e74ull,
      .op_hash_lo = 0x7374617465626974ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = WindowSource();
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(
      adapter, 8u, sizeof(VulkanWindowParams), plan, artifact);
}

[[nodiscard]] VulkanCollectivePipeline *
AcquireGatePipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x7265736964656e74ull,
      .op_hash_lo = 0x6761746562697431ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = GateSource();
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter, 3u, sizeof(GateParams), plan,
                                         artifact);
}

void EncodeDispatchBarrier(VkCommandBuffer command) noexcept;

[[nodiscard]] bool GateIndirect(void *const context,
                                VulkanDispatchCapture &capture,
                                const VkCommandBuffer command,
                                const VkBuffer source,
                                const VkDeviceSize offset) noexcept {
  auto *const resources = static_cast<VulkanWindowResources *>(context);
  if (resources == nullptr || resources->adapter == nullptr ||
      resources->gate_pipeline == nullptr || command == VK_NULL_HANDLE ||
      source == VK_NULL_HANDLE || (offset & 3u) != 0u ||
      capture.mapped == nullptr || capture.original == nullptr ||
      capture.owners == nullptr || capture.arguments == VK_NULL_HANDLE ||
      capture.cursor >= capture.capacity ||
      capture.pipeline == VK_NULL_HANDLE || capture.layout == VK_NULL_HANDLE ||
      capture.descriptor == VK_NULL_HANDLE ||
      resources->adapter->storage_align == 0u) {
    return false;
  }
  const VkDeviceSize base =
      offset - (offset % resources->adapter->storage_align);
  if (offset > std::numeric_limits<VkDeviceSize>::max() -
                   sizeof(VkDispatchIndirectCommand)) {
    return false;
  }
  const VkDeviceSize end = offset + sizeof(VkDispatchIndirectCommand);
  const VkDeviceSize range = end - base;
  if (range == 0u || range > resources->adapter->storage_limit) {
    return false;
  }
  try {
    resources->gates.push_back(VulkanGateRoute{
        .source =
            VulkanBuffer{
                .buffer = source,
                .bytes = end,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            },
    });
  } catch (const std::bad_alloc &) {
    return false;
  }
  VulkanGateRoute &route = resources->gates.back();
  if (!AcquireVulkanCollectiveDescriptorSet(*resources->adapter,
                                            *resources->gate_pipeline, 3u,
                                            route.descriptor)) {
    resources->gates.pop_back();
    return false;
  }
  const std::array<VulkanStorageBinding, 3u> bindings{
      VulkanStorageBinding{&route.source, base, range},
      VulkanStorageBindingFor(resources->arguments),
      VulkanStorageBindingFor(resources->states),
  };
  if (!WriteVulkanStorageDescriptorSet(*resources->adapter, route.descriptor,
                                       bindings)) {
    resources->gates.pop_back();
    return false;
  }
  const std::size_t slot = capture.cursor++;
  capture.mapped[slot] = {};
  capture.original[slot] = {};
  capture.owners[slot] = std::numeric_limits<std::uint32_t>::max();
  const GateParams params{
      .state = capture.owner,
      .source_word = static_cast<std::uint32_t>((offset - base) / 4u),
      .target_word = static_cast<std::uint32_t>(slot * 3u),
  };
  EncodeVulkanComputeToComputeBarrier(command);
  ::vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      resources->gate_pipeline->pipeline);
  ::vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources->gate_pipeline->pipeline_layout, 0u, 1u,
                            &route.descriptor, 0u, nullptr);
  ::vkCmdPushConstants(command, resources->gate_pipeline->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                       &params);
  ::vkCmdDispatch(command, 1u, 1u, 1u);
  EncodeDispatchBarrier(command);
  ::vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      capture.pipeline);
  ::vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            capture.layout, 0u, 1u, &capture.descriptor, 0u,
                            nullptr);
  if (capture.has_push) {
    ::vkCmdPushConstants(command, capture.layout, capture.push_stages,
                         capture.push_offset, capture.push_size,
                         capture.push.data() + capture.push_offset);
  }
  ::vkCmdDispatchIndirect(
      command, capture.arguments,
      static_cast<VkDeviceSize>(slot * sizeof(VkDispatchIndirectCommand)));
  ++capture.indirect_count;
  return true;
}

[[nodiscard]] const char *DescriptorFailure(VulkanAdapter &adapter) noexcept {
  const char *const reason = VulkanLastError(&adapter);
  return reason == nullptr || reason[0] == '\0'
             ? "accel_vulkan_descriptor_unavailable"
             : reason;
}

[[nodiscard]] bool SameBinding(const VulkanStorageBinding &left,
                               const VulkanStorageBinding &right) noexcept {
  return left.buffer == right.buffer && left.offset == right.offset &&
         left.range == right.range;
}

void EncodeDispatchBarrier(const VkCommandBuffer command) noexcept {
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
}

} // namespace

rund::AccelCheck PrepareVulkanWindow(
    VulkanAdapter &adapter, const std::span<const BackendBatchEntry> entries,
    const std::uint64_t dispatch_capacity,
    const PreparedPipelineStatusLayout &status, const VulkanBuffer &control,
    VulkanWindowResources &resources) {
  resources = {};
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
    return rund::AccelCheck{true, "ok"};
  }
  if (dispatch_capacity == 0u || control.buffer == VK_NULL_HANDLE ||
      control.bytes < sizeof(PreparedPipelineControl) ||
      dispatch_capacity > std::numeric_limits<std::uint32_t>::max() ||
      dispatch_capacity > std::numeric_limits<std::uint32_t>::max() / 3u ||
      dispatch_capacity > std::numeric_limits<std::size_t>::max() /
                              sizeof(VkDispatchIndirectCommand) ||
      dispatch_capacity >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.adapter = &adapter;
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
  try {
    resources.original.resize(static_cast<std::size_t>(dispatch_capacity));
    resources.routes.reserve(route_count);
    resources.descriptor_leases.reserve(state_count);
  } catch (const std::bad_alloc &) {
    DestroyVulkanWindow(resources);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
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
  resources.capture.indirect = &GateIndirect;

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
      const bool nested = window->nested();
      const bool nested_shape_valid =
          !nested || (window->outer_bound != 0u &&
                      window->outer_iteration < window->outer_bound &&
                      window->inner_bound != 0u &&
                      ((window->phase == BackendWindowPhase::NestedSeed &&
                        window->route == 0u) ||
                       (window->phase == BackendWindowPhase::NestedAction &&
                        window->inner_iteration < window->inner_bound &&
                        window->route == 0u) ||
                       (window->phase == BackendWindowPhase::NestedFold &&
                        window->route < 3u)));
      const std::uint32_t template_index = entries[entry_index].template_index;
      VulkanResidentBufferResult count = ResolveVulkanResidentBuffer(
          resident, window->count.source, window->count.handle,
          "compute_resident_id_invalid", true);
      VulkanCopyRange count_range{};
      const bool window_valid =
          count.check.ok && count.device_buffer != nullptr &&
          window->maximum != 0u && window->tile != 0u &&
          window->tile <= window->maximum && window->bound != 0u &&
          window->iteration < window->bound && nested_shape_valid &&
          template_index < status.active_step_count &&
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
                  .iteration = window->iteration,
                  .expected = window->expected,
                  .state = window->state,
                  .has_terminal =
                      static_cast<std::uint32_t>(window->has_terminal),
                  .command_count =
                      static_cast<std::uint32_t>(dispatch_capacity),
                  .phase = static_cast<std::uint32_t>(window->phase),
                  .declared_step = status.declared_steps[template_index],
                  .overflow_reason = static_cast<std::uint32_t>(
                      rund::compute::Reason::BoundedCountInvalid),
                  .inner_bound = window->inner_bound,
              },
          .entry = static_cast<std::uint32_t>(entry_index),
      });
    }
  }

  bool ready = false;
  try {
    std::vector<std::size_t> descriptor_owners(
        state_count, std::numeric_limits<std::size_t>::max());
    VulkanLeaseScope lease_scope{adapter, resources.descriptor_leases};
    resources.pipeline = AcquireWindowPipeline(adapter);
    resources.gate_pipeline = AcquireGatePipeline(adapter);
    ready = resources.pipeline != nullptr && resources.gate_pipeline != nullptr;
    for (std::size_t index = 0u; ready && index < resources.routes.size();
         ++index) {
      VulkanWindowRoute &route = resources.routes[index];
      if (route.params.state >= descriptor_owners.size()) {
        ready = false;
        break;
      }
      const std::size_t owner = descriptor_owners[route.params.state];
      if (owner != std::numeric_limits<std::size_t>::max()) {
        const VulkanWindowRoute &first = resources.routes[owner];
        bool same = SameBinding(route.count_binding, first.count_binding);
        for (std::size_t bank = 0u;
             same && bank < route.terminal_bindings.size(); ++bank) {
          same = SameBinding(route.terminal_bindings[bank],
                             first.terminal_bindings[bank]);
        }
        if (!same || first.descriptor == VK_NULL_HANDLE) {
          ready = false;
          break;
        }
        route.descriptor = first.descriptor;
        continue;
      }
      ready = ready && AcquireVulkanCollectiveDescriptorSet(
                           adapter, *resources.pipeline, 8u, route.descriptor);
      if (!ready) {
        break;
      }
      const std::array<VulkanStorageBinding, 8u> bindings{
          route.terminal_bindings[0],
          route.terminal_bindings[1],
          route.terminal_bindings[2],
          route.count_binding,
          VulkanStorageBindingFor(resources.states),
          VulkanStorageBindingFor(resources.arguments),
          VulkanStorageBindingFor(resources.owners),
          VulkanStorageBindingFor(control)};
      ready =
          WriteVulkanStorageDescriptorSet(adapter, route.descriptor, bindings);
      if (!ready) {
        break;
      }
      descriptor_owners[route.params.state] = index;
    }
  } catch (const std::bad_alloc &) {
    DestroyVulkanWindow(resources);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!ready) {
    const char *const reason = DescriptorFailure(adapter);
    DestroyVulkanWindow(resources);
    return rund::AccelCheck{false, reason};
  }
  return rund::AccelCheck{true, "ok"};
}

void DestroyVulkanWindow(VulkanWindowResources &resources) noexcept {
  if (resources.adapter != nullptr) {
    ReleaseVulkanLeases(resources.descriptor_leases);
    DestroyVulkanBuffer(*resources.adapter, resources.states);
    DestroyVulkanBuffer(*resources.adapter, resources.arguments);
    DestroyVulkanBuffer(*resources.adapter, resources.original_arguments);
    DestroyVulkanBuffer(*resources.adapter, resources.owners);
  }
  resources = {};
}

bool EncodeVulkanWindow(const VkCommandBuffer command,
                        const VulkanWindowResources &resources,
                        const std::uint32_t entry,
                        const bool preflight) noexcept {
  const auto begin = std::lower_bound(
      resources.routes.begin(), resources.routes.end(), entry,
      [](const VulkanWindowRoute &route, const std::uint32_t value) {
        return route.entry < value;
      });
  const auto end = std::upper_bound(
      begin, resources.routes.end(), entry,
      [](const std::uint32_t value, const VulkanWindowRoute &route) {
        return value < route.entry;
      });
  if (begin == end) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (auto route = begin; route != end; ++route) {
    const auto phase = static_cast<BackendWindowPhase>(route->params.phase);
    if (route->descriptor == VK_NULL_HANDLE ||
        (preflight && phase != BackendWindowPhase::NestedSeed)) {
      return false;
    }
    VulkanWindowParams params = route->params;
    if (preflight) {
      params.phase |= 0x80000000u;
    }
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route->descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                        &params);
    ::vkCmdDispatch(command, 1u, 1u, 1u);
  }
  EncodeDispatchBarrier(command);
  return true;
}

bool EncodeVulkanWindowStart(const VkCommandBuffer command,
                             const VulkanWindowResources &resources) noexcept {
  if (command == VK_NULL_HANDLE || resources.states.buffer == VK_NULL_HANDLE ||
      resources.arguments.buffer == VK_NULL_HANDLE ||
      resources.original_arguments.buffer == VK_NULL_HANDLE ||
      resources.states.bytes == 0u || resources.arguments.bytes == 0u ||
      resources.arguments.bytes != resources.original_arguments.bytes) {
    return false;
  }
  vkCmdFillBuffer(command, resources.states.buffer, 0u, resources.states.bytes,
                  0u);
  const VkBufferCopy copy{.size = resources.arguments.bytes};
  vkCmdCopyBuffer(command, resources.original_arguments.buffer,
                  resources.arguments.buffer, 1u, &copy);
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
  return true;
}

bool FreezeVulkanWindow(VulkanWindowResources &resources) noexcept {
  const std::uint64_t argument_bytes =
      static_cast<std::uint64_t>(resources.original.size()) *
      sizeof(VkDispatchIndirectCommand);
  return resources.state_count == 0u ||
         (argument_bytes == resources.original_arguments.bytes &&
          UploadVulkanBuffer(resources.original_arguments,
                             resources.original.data(), argument_bytes));
}

std::uint64_t
VulkanWindowHostBytes(const VulkanWindowResources &resources) noexcept {
  const std::uint64_t routes = capacity_bytes(resources.routes);
  const std::uint64_t leases = capacity_bytes(resources.descriptor_leases);
  const std::uint64_t original = capacity_bytes(resources.original);
  const std::uint64_t gates = capacity_bytes(resources.gates);
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t base =
      routes > maximum - leases ? maximum : routes + leases;
  const std::uint64_t captured =
      base > maximum - original ? maximum : base + original;
  return captured > maximum - gates ? maximum : captured + gates;
}

} // namespace rund::node::accel::detail

#endif
