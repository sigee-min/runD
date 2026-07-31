#pragma once

#include <accel/device.hpp>

#include "../../reason.hpp"
#include "../../vulkan/local.hpp"
#include "../local.hpp"
#include "test/assert.hpp"

#include <string_view>

namespace node_accel_contract {

[[nodiscard]] bool
MetalRepeatedStagedRunsReportWarmRuntimeStats(const rund::AccelDevice &pick);
[[nodiscard]] bool
MetalResidentBufferRegistryValidatesPrivateRefs(const rund::AccelDevice &pick);
[[nodiscard]] bool
RequiredMetalRunsAccelGraphScan(const rund::AccelDevice &pick);
int RunAccelKernelCollectiveSurfaceContract();
[[nodiscard]] bool PublicBufferApiContract(const rund::AccelDevice &pick);
[[nodiscard]] bool
ResidentWindowCollapseContract(const rund::AccelDevice &pick);
[[nodiscard]] bool
RequiredVulkanRunsAccelGraphScan(const rund::AccelDevice &pick);
[[nodiscard]] bool
AvailableBackendsRunSameAccelGraphSort(const rund::AccelDevice &metal,
                                       const rund::AccelDevice &vulkan);
[[nodiscard]] bool
RequiredVulkanPickIsPlatformAware(const rund::AccelDevice &pick);
[[nodiscard]] bool VulkanPickHasDiscoveryCaps(const rund::AccelDevice &pick);
[[nodiscard]] bool VulkanDirectBackendLastErrorIsPrecise();
[[nodiscard]] bool VulkanRepeatedStagedRunsReportWarmRuntimeStats();
[[nodiscard]] bool RuntimePolicyChoosesOnlyFromLocalEvidence();
[[nodiscard]] bool PublicContextApiContract();
[[nodiscard]] bool ContextRejectsForgedPickOwnerInSupportPaths();
[[nodiscard]] bool AccelGraphKernelCompileContract();
[[nodiscard]] bool AccelGraphKernelResidentRunContract();
[[nodiscard]] bool KernelBindingIndicesUseInlineStorageUntilOverflow();
[[nodiscard]] bool
BackendRejectsPlanBeyondFrozenCaps(const rund::AccelDevice &pick);
[[nodiscard]] bool
BackendRejectsUnderreportedStaging(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendAcceptsSimpleWindow(const rund::AccelDevice &pick);
[[nodiscard]] bool
BackendRejectsPlanObligationMismatches(const rund::AccelDevice &pick);
int RunAccelCoreContracts();
int RunAccelMetalContracts(const rund::AccelDevice &pick);
int RunAccelVulkanContracts(const rund::AccelDevice &metal_pick,
                            const rund::AccelDevice &vulkan_pick);
int RunAccelPickContracts();

} // namespace node_accel_contract
