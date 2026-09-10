#pragma once

#include "../../manifest.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool BuildVulkanMapManifest(const KernelExecutionStep &,
                                          const rund::kernel::ComputePlan &,
                                          const BoundStep *, bool has_checks,
                                          bool controlled,
                                          PreparedBackendManifest &) noexcept;

[[nodiscard]] bool BuildVulkanScanManifest(const KernelExecutionStep &,
                                           const rund::kernel::ComputePlan &,
                                           PreparedBackendManifest &) noexcept;

[[nodiscard]] bool
BuildVulkanReductionManifest(const KernelExecutionStep &,
                             const rund::kernel::ComputePlan &,
                             PreparedBackendManifest &) noexcept;

[[nodiscard]] bool BuildVulkanSortManifest(const KernelExecutionStep &,
                                           const rund::kernel::ComputePlan &,
                                           PreparedBackendManifest &) noexcept;

[[nodiscard]] bool
BuildVulkanCollectiveManifest(const KernelExecutionStep &,
                              PreparedBackendManifest &) noexcept;

[[nodiscard]] bool
BuildVulkanScatterManifest(const KernelExecutionStep &,
                           PreparedBackendManifest &) noexcept;

[[nodiscard]] bool BuildVulkanRangeManifest(const KernelExecutionStep &,
                                            PreparedBackendManifest &) noexcept;

[[nodiscard]] bool
BuildVulkanNumericManifest(const KernelExecutionStep &,
                           PreparedBackendManifest &) noexcept;

#endif

} // namespace rund::node::accel::detail
