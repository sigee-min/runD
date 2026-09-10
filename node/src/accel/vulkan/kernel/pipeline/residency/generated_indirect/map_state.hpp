#pragma once

#include "../../../../map/local.hpp"

#include <cstdint>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] inline bool same_buffer(const VulkanBuffer &left,
                                      const VulkanBuffer &right) noexcept {
  return left.buffer == right.buffer && left.bytes == right.bytes &&
         left.offset == right.offset && left.usage == right.usage;
}

[[nodiscard]] inline bool
map_owner_matches(const VulkanMapEncodeResources &map,
                  const VulkanCollectivePipeline *const pipeline,
                  const VkDescriptorSet descriptor, const VulkanBuffer &gate,
                  const VulkanBuffer &summary) noexcept {
  return map.generated_control_pipeline == pipeline &&
         map.generated_control_descriptor == descriptor &&
         same_buffer(map.generated_gate_binding, gate) &&
         same_buffer(map.generated_summary_binding, summary);
}

struct GeneratedMapState final {
  VulkanCollectivePipeline *pipeline{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  VulkanBuffer gate{};
  VulkanBuffer summary{};
  std::uint64_t owner{};
  std::uint64_t plan{};
  std::uint64_t token{};
  std::uint64_t run{};
  std::uint32_t slot{};
  std::uint32_t stride{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint64_t first_descriptor_generation{};
  std::uint64_t descriptor_generation_stride{};
  VulkanMapMode mode{VulkanMapMode::Direct};
};

class MapModeGuard final {
public:
  explicit MapModeGuard(VulkanMapEncodeResources &map) noexcept
      : map_(map), saved_(map.mode) {
    map_.mode = VulkanMapMode::Generated;
  }

  ~MapModeGuard() { map_.mode = saved_; }

  MapModeGuard(const MapModeGuard &) = delete;
  MapModeGuard &operator=(const MapModeGuard &) = delete;

private:
  VulkanMapEncodeResources &map_;
  VulkanMapMode saved_;
};

[[nodiscard]] inline GeneratedMapState
save_map_state(const VulkanMapEncodeResources &map) noexcept {
  return GeneratedMapState{
      .pipeline = map.generated_control_pipeline,
      .descriptor = map.generated_control_descriptor,
      .gate = map.generated_gate_binding,
      .summary = map.generated_summary_binding,
      .owner = map.generated_owner,
      .plan = map.generated_plan,
      .token = map.generated_token,
      .run = map.generated_run,
      .slot = map.generated_slot,
      .stride = map.generated_stride,
      .first_control_generation = map.generated_first_control_generation,
      .control_generation_stride = map.generated_control_generation_stride,
      .first_descriptor_generation = map.generated_first_descriptor_generation,
      .descriptor_generation_stride =
          map.generated_descriptor_generation_stride,
      .mode = map.mode};
}

inline void restore_map_state(VulkanMapEncodeResources &map,
                              const GeneratedMapState &state) noexcept {
  map.generated_control_pipeline = state.pipeline;
  map.generated_control_descriptor = state.descriptor;
  map.generated_gate_binding = state.gate;
  map.generated_summary_binding = state.summary;
  map.generated_owner = state.owner;
  map.generated_plan = state.plan;
  map.generated_token = state.token;
  map.generated_run = state.run;
  map.generated_slot = state.slot;
  map.generated_stride = state.stride;
  map.generated_first_control_generation = state.first_control_generation;
  map.generated_control_generation_stride = state.control_generation_stride;
  map.generated_first_descriptor_generation = state.first_descriptor_generation;
  map.generated_descriptor_generation_stride =
      state.descriptor_generation_stride;
  map.mode = state.mode;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
