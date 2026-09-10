#include "src/accel/kernel/backend/execute.hpp"
#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/vulkan/kernel/control.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/accel/vulkan/kernel/publish.hpp"
#include "src/accel/vulkan/kernel/window.hpp"

#include <cstdint>
#include <limits>
#include <string_view>

#include "../manifest.hpp"

namespace node_accel_contract {

[[nodiscard]] bool VulkanPhysicalCommandShapeIsNonOverlapping() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation reservation{
      .window_count = 3u,
      .backend_dispatch_count = 7u,
      .backend_reset_dispatch_count = 3u,
      .backend_window_dispatch_count = 7u,
      .backend_indirect_dispatch_count = 2u,
      .backend_window_state_count = 1u,
      .backend_window_descriptor_state_count = 1u,
      .backend_status_source_count = 1u,
      .backend_status_entry_count = 1u,
      .backend_telemetry_count = 1u,
      .backend_status_command_count = 4u,
      .backend_telemetry_command_count = 2u,
      .backend_publication_count = 2u,
      .backend_terminal_publication_count = 1u,
      .backend_publication_command_count = 5u,
      .backend_parameter_bytes = 13u,
  };
  constexpr std::uint64_t expected_window_control = 6u;
  constexpr std::uint64_t expected_commands =
      7u + 3u + 4u + 2u + expected_window_control + 2u + 5u + 3u;
  constexpr std::uint64_t expected_parameters =
      13u + 2u * sizeof(VulkanPipelineTelemetryParams) +
      expected_window_control * sizeof(VulkanWindowParams) +
      2u * VulkanGateParameterBytes + 5u * sizeof(VulkanPipelinePublishParams) +
      2u * VulkanPipelineControlParameterBytes;
  const rund::AccelCheck planned =
      PlanVulkanPipelineStructure(rund::AccelContext{}, reservation);
  return planned.ok &&
         reservation.backend_window_control_command_count ==
             expected_window_control &&
         reservation.backend_command_count == expected_commands &&
         reservation.backend_parameter_bytes == expected_parameters;
#else
  return true;
#endif
}

[[nodiscard]] bool VulkanPhysicalCapacityFailureIsTransactional() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const auto rejected_without_commit = [](auto reservation) {
    const PreparedKernelPipelineReservation before = reservation;
    const rund::AccelCheck planned =
        PlanVulkanPipelineStructure(rund::AccelContext{}, reservation);
    return !planned.ok && planned.reason != nullptr &&
           std::string_view{planned.reason} == "compute_pipeline_capacity" &&
           reservation == before;
  };

  PreparedKernelPipelineReservation window_product{
      .host_bytes = 17u,
      .native_bytes = 19u,
      .window_count = maximum / 2u + 1u,
      .backend_window_dispatch_count = 1u,
      .backend_window_state_count = 1u,
      .backend_window_descriptor_state_count = 1u,
  };
  PreparedKernelPipelineReservation command_sum{
      .host_bytes = 23u,
      .native_bytes = 29u,
      .backend_dispatch_count = maximum,
  };
  PreparedKernelPipelineReservation parameter_product{
      .host_bytes = 37u,
      .native_bytes = 41u,
      .backend_telemetry_command_count =
          maximum / sizeof(VulkanPipelineTelemetryParams) + 1u,
      .backend_parameter_bytes = 31u,
  };
  return rejected_without_commit(window_product) &&
         rejected_without_commit(command_sum) &&
         rejected_without_commit(parameter_product);
#else
  return true;
#endif
}

} // namespace node_accel_contract
