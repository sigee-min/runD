#pragma once

#include "../control.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <array>
#include <cstdint>

namespace rund::node::accel::detail {

inline constexpr std::uint32_t VulkanPipelineControlThreads = 256u;

struct VulkanCanonicalStatusParams final {
  std::uint32_t first{};
  std::uint32_t count{};
  std::uint32_t rule{};
  std::uint32_t success{};
  std::uint32_t mapping_count{};
  std::uint32_t invalid_reason{};
  std::array<std::uint32_t, 8u> raw_values{};
  std::array<std::uint32_t, 8u> reasons{};
};

static_assert(sizeof(VulkanCanonicalStatusParams) ==
              VulkanPipelineCanonicalParameterBytes);

struct VulkanPipelineControlParams final {
  std::uint32_t first{};
  std::uint32_t count{};
  std::uint32_t declared_step{};
  std::uint32_t phase{};
  std::uint32_t failed_outer_window{PreparedPipelineNoStep};
  std::uint32_t failed_inner_iteration{PreparedPipelineNoStep};
  std::uint32_t failed_nested_phase{PipelineNestedPhaseNoneCode};
  std::uint32_t reserved{};
  std::uint32_t generation_stride{};
  std::uint32_t declared_step_count{};
};

static_assert(sizeof(VulkanPipelineControlParams) ==
              VulkanPipelineControlParameterBytes);

} // namespace rund::node::accel::detail

#endif
