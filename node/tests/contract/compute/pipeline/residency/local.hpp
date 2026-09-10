#pragma once

#include "authority/local.hpp"

namespace rund_node_test_pipeline_residency {

[[nodiscard]] int CheckPlanner();
[[nodiscard]] int CheckIdentity();
[[nodiscard]] int CheckGraphPlanner();
[[nodiscard]] int CheckGraphProjection();
[[nodiscard]] int CheckGraphForecast();
[[nodiscard]] int CheckGraphPromote();
[[nodiscard]] int CheckGraphHostInputLayout();
[[nodiscard]] int CheckGraphDrain();
[[nodiscard]] int CheckGraphPersist();
[[nodiscard]] int CheckGraphWavefront();
[[nodiscard]] int CheckHostRingCapacities();
[[nodiscard]] int CheckFootprints();
[[nodiscard]] int CheckCycle();
[[nodiscard]] int CheckExecution();
[[nodiscard]] int CheckFetchFill();
[[nodiscard]] int CheckSlidingModel();
[[nodiscard]] int CheckSlidingAuthority();
[[nodiscard]] int CheckSlidingGeneration();
[[nodiscard]] int CheckNativeSliding();
[[nodiscard]] int CheckPersistentSliding();
[[nodiscard]] int CheckServiceFreeDirect();
[[nodiscard]] int CheckDeviceVsm();
[[nodiscard]] int CheckWindow();
[[nodiscard]] int CheckPoolLending();
[[nodiscard]] int CheckPoolExtentAdmission();
[[nodiscard]] int CheckPhysicalBufferViews();

} // namespace rund_node_test_pipeline_residency

namespace rund_node_test_pipeline {
[[nodiscard]] int CheckVulkanResidencyWindow();
}
