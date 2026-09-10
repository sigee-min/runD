#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/backend/execute.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/publication.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

#include "hooks.hpp"
#include "manifest.hpp"

namespace node_accel_contract {

[[nodiscard]] bool PreparedBackendProjectionIsRouteWiseAndChecked() {
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation projection{};
  const PreparedKernelRouteReservation first{
      .route_step_count = 1u,
      .host_transient_bytes = 11u,
      .dispatch_count = 2u,
      .capture_direct_dispatch_count = 3u,
      .capture_indirect_dispatch_count = 1u,
      .capture_binding_slot_upper = 6u,
      .reset_dispatch_count = 1u,
      .status_entry_count = 4u,
      .status_source_count = 2u,
      .status_command_count = 3u,
      .status_parameter_bytes = 5u,
      .telemetry_source_count = 1u,
  };
  const PreparedKernelRouteReservation second{
      .route_step_count = 3u,
      .host_transient_bytes = 13u,
      .dispatch_count = 5u,
      .capture_direct_dispatch_count = 4u,
      .capture_indirect_dispatch_count = 2u,
      .capture_binding_slot_upper = 11u,
      .reset_dispatch_count = 4u,
      .status_entry_count = 7u,
      .status_source_count = 3u,
      .status_command_count = 4u,
      .status_parameter_bytes = 7u,
      .telemetry_source_count = 2u,
  };
  if (!AccumulatePreparedKernelRouteProjectionForContract(projection, first, 2u,
                                                          2u, 2u) ||
      !AccumulatePreparedKernelRouteProjectionForContract(projection, second,
                                                          3u, 3u, 1u) ||
      projection.occurrence_count != 5u ||
      projection.host_transient_bytes != 24u ||
      projection.backend_dispatch_count != 19u ||
      projection.backend_reset_dispatch_count != 14u ||
      projection.backend_step_occurrence_count != 11u ||
      projection.backend_step_description_count != 11u ||
      projection.backend_status_entry_count != 29u ||
      projection.backend_window_dispatch_count != 14u ||
      projection.backend_indirect_dispatch_count != 4u ||
      projection.backend_command_binding_slot_upper != 11u ||
      projection.backend_status_source_count != 13u ||
      projection.backend_telemetry_count != 8u ||
      projection.backend_status_command_count != 18u ||
      projection.backend_telemetry_command_count != 8u ||
      projection.backend_parameter_bytes != 31u) {
    return false;
  }

  const PreparedKernelPipelineReservation before = projection;
  const auto unchanged = [&before](
                             const PreparedKernelPipelineReservation &value) {
    return value.occurrence_count == before.occurrence_count &&
           value.host_transient_bytes == before.host_transient_bytes &&
           value.backend_dispatch_count == before.backend_dispatch_count &&
           value.backend_reset_dispatch_count ==
               before.backend_reset_dispatch_count &&
           value.backend_step_occurrence_count ==
               before.backend_step_occurrence_count &&
           value.backend_step_description_count ==
               before.backend_step_description_count &&
           value.backend_status_entry_count ==
               before.backend_status_entry_count &&
           value.backend_command_binding_slot_upper ==
               before.backend_command_binding_slot_upper &&
           value.backend_status_source_count ==
               before.backend_status_source_count &&
           value.backend_telemetry_count == before.backend_telemetry_count &&
           value.backend_status_command_count ==
               before.backend_status_command_count &&
           value.backend_telemetry_command_count ==
               before.backend_telemetry_command_count &&
           value.backend_parameter_bytes == before.backend_parameter_bytes &&
           value.backend_window_dispatch_count ==
               before.backend_window_dispatch_count &&
           value.backend_indirect_dispatch_count ==
               before.backend_indirect_dispatch_count &&
           value.backend_window_control_command_count ==
               before.backend_window_control_command_count &&
           value.backend_publication_command_count ==
               before.backend_publication_command_count;
  };
  const PreparedKernelRouteReservation overflow{
      .route_step_count = 1u,
      .dispatch_count = std::numeric_limits<std::uint64_t>::max(),
      .status_entry_count = 1u,
  };
  if (AccumulatePreparedKernelRouteProjectionForContract(projection, overflow,
                                                         1u, 2u, 0u) ||
      !unchanged(projection)) {
    return false;
  }
  const std::array overflow_routes{
      PreparedKernelRouteReservation{
          .route_step_count = std::numeric_limits<std::uint64_t>::max(),
          .dispatch_count = 1u,
      },
      PreparedKernelRouteReservation{
          .route_step_count = 1u,
          .dispatch_count = 1u,
          .status_parameter_bytes = std::numeric_limits<std::uint64_t>::max(),
      },
      PreparedKernelRouteReservation{
          .route_step_count = 1u,
          .dispatch_count = 1u,
          .capture_direct_dispatch_count =
              std::numeric_limits<std::uint64_t>::max(),
          .capture_indirect_dispatch_count = 1u,
      },
      PreparedKernelRouteReservation{
          .route_step_count = 1u,
          .dispatch_count = 1u,
          .capture_direct_dispatch_count =
              std::numeric_limits<std::uint64_t>::max(),
      },
  };
  const std::array<std::uint64_t, 4u> compact_counts{2u, 1u, 1u, 1u};
  const std::array<std::uint64_t, 4u> occurrence_counts{1u, 2u, 1u, 1u};
  const std::array<std::uint64_t, 4u> window_counts{0u, 0u, 1u, 2u};
  for (std::size_t index = 0u; index < overflow_routes.size(); ++index) {
    if (AccumulatePreparedKernelRouteProjectionForContract(
            projection, overflow_routes[index], compact_counts[index],
            occurrence_counts[index], window_counts[index]) ||
        !unchanged(projection)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool PreparedPublicationCommandShapeIsCanonical() {
  using namespace rund::node::accel::detail;
  std::uint64_t commands = 0u;
  if (!PreparedKernelPublicationCommandContribution(
          PreparedKernelPublicationKind::Terminal, 0u, commands) ||
      commands != 2u ||
      !PreparedKernelPublicationCommandContribution(
          PreparedKernelPublicationKind::Window, 63u, commands) ||
      commands != 63u ||
      !PreparedKernelPublicationCommandContribution(
          PreparedKernelPublicationKind::Window, 4u, commands) ||
      commands != 4u) {
    return false;
  }
  return !PreparedKernelPublicationCommandContribution(
             PreparedKernelPublicationKind::Terminal, 1u, commands) &&
         !PreparedKernelPublicationCommandContribution(
             PreparedKernelPublicationKind::Window, 0u, commands) &&
         !PreparedKernelPublicationCommandContribution(
             static_cast<PreparedKernelPublicationKind>(0xffu), 1u, commands);
}

[[nodiscard]] bool PreparedBackendControlManifestIsDimensionallyClosed() {
  using namespace rund::node::accel::detail;
  PreparedBackendManifest manifest{};
  if (!ValidPreparedBackendControlManifest(manifest)) {
    return false;
  }
  manifest.status_source_count = 1u;
  if (ValidPreparedBackendControlManifest(manifest)) {
    return false;
  }
  manifest.status_entry_count = 4u;
  manifest.status_command_count = 2u;
  manifest.status_parameter_bytes = 16u;
  if (!ValidPreparedBackendControlManifest(manifest)) {
    return false;
  }
  manifest.status_entry_count = 0u;
  return !ValidPreparedBackendControlManifest(manifest);
}

} // namespace node_accel_contract
