#pragma once

#include "../../../../kernel/residency/schedule.hpp"
#include <kernel/program/compute/graph/schema.hpp>

#include "model.hpp"

namespace rund::node::accel::detail::vulkan_residency_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] constexpr VulkanResidencyMode
mode(const VulkanResidencyAdmissionCandidate candidate) noexcept {
  switch (candidate) {
  case VulkanResidencyAdmissionCandidate::GraphDirect:
    return VulkanResidencyMode::GraphStageDirect;
  case VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect:
    return VulkanResidencyMode::GraphStageGeneratedIndirect;
  case VulkanResidencyAdmissionCandidate::GraphStageSequence:
    return VulkanResidencyMode::GraphStageSequence;
  case VulkanResidencyAdmissionCandidate::GeneratedIndirect:
    return VulkanResidencyMode::GeneratedIndirectMap;
  case VulkanResidencyAdmissionCandidate::Direct:
    return VulkanResidencyMode::Direct;
  }
  return VulkanResidencyMode::Direct;
}

[[nodiscard]] constexpr bool owns(const VulkanResidencyMode value) noexcept {
  return value == VulkanResidencyMode::GraphStageGeneratedIndirect ||
         value == VulkanResidencyMode::GraphStageSequence;
}

[[nodiscard]] constexpr bool is_map(const VulkanResidencyMode value) noexcept {
  return value == VulkanResidencyMode::GeneratedIndirectMap;
}

[[nodiscard]] constexpr bool
is_direct(const VulkanResidencyMode value) noexcept {
  return value == VulkanResidencyMode::Direct;
}

[[nodiscard]] constexpr bool
is_graph_direct(const VulkanResidencyMode value) noexcept {
  return value == VulkanResidencyMode::GraphStageDirect;
}

[[nodiscard]] constexpr bool
is_graph(const VulkanResidencyMode value) noexcept {
  return value == VulkanResidencyMode::GraphStageDirect ||
         value == VulkanResidencyMode::GraphStageGeneratedIndirect ||
         value == VulkanResidencyMode::GraphStageSequence;
}

[[nodiscard]] constexpr bool
recognized(const VulkanResidencyMode value) noexcept {
  return is_map(value) || is_direct(value) || is_graph(value);
}

#endif

} // namespace rund::node::accel::detail::vulkan_residency_detail
