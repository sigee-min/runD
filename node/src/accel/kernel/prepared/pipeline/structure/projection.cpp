#include "../reservation.hpp"
#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

// Projects canonical, unfused route occurrences into one backend stream.
// `dispatch_count` is the backend's physical body/view-command upper; reset and
// control families remain independent below. Nested transducers may materialize
// fewer physical commands, but this sum never substitutes that later
// optimization for the authored route ownership proved here.
[[nodiscard]] bool
accumulate_route_projection(PreparedKernelPipelineReservation &projection,
                            const PreparedKernelRouteReservation &route,
                            const std::uint64_t compact_entry_count,
                            const std::uint64_t occurrence_count,
                            const std::uint64_t window_count) noexcept {
  if (route.route_step_count == 0u || route.dispatch_count == 0u ||
      compact_entry_count == 0u) {
    return false;
  }
  PreparedKernelPipelineReservation candidate = projection;
  candidate.backend_command_binding_slot_upper =
      std::max(candidate.backend_command_binding_slot_upper,
               route.capture_binding_slot_upper);
  std::uint64_t contribution = 0u;
  std::uint64_t capture_dispatch_count = 0u;
  if (!accumulate(candidate.occurrence_count, occurrence_count) ||
      !accumulate(candidate.host_transient_bytes, route.host_transient_bytes) ||
      !mul(route.dispatch_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_dispatch_count, contribution) ||
      !mul(route.reset_dispatch_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_reset_dispatch_count, contribution) ||
      !mul(route.route_step_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_step_occurrence_count, contribution) ||
      !mul(route.route_step_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_step_description_count, contribution) ||
      !mul(route.status_entry_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_status_entry_count, contribution) ||
      !mul(route.status_source_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_status_source_count, contribution) ||
      !mul(route.telemetry_source_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_telemetry_count, contribution) ||
      !mul(route.status_command_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_status_command_count, contribution) ||
      !mul(route.telemetry_source_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_telemetry_command_count, contribution) ||
      !mul(route.status_parameter_bytes, occurrence_count, contribution) ||
      !accumulate(candidate.backend_parameter_bytes, contribution) ||
      !add(route.capture_direct_dispatch_count,
           route.capture_indirect_dispatch_count, capture_dispatch_count) ||
      !mul(capture_dispatch_count, window_count, contribution) ||
      !accumulate(candidate.backend_window_dispatch_count, contribution) ||
      !mul(route.capture_indirect_dispatch_count, window_count, contribution) ||
      !accumulate(candidate.backend_indirect_dispatch_count, contribution)) {
    return false;
  }
  projection = candidate;
  return true;
}

bool AccumulatePreparedKernelRouteProjectionForContract(
    PreparedKernelPipelineReservation &projection,
    const PreparedKernelRouteReservation &route,
    const std::uint64_t compact_entry_count,
    const std::uint64_t occurrence_count,
    const std::uint64_t window_count) noexcept {
  return accumulate_route_projection(projection, route, compact_entry_count,
                                     occurrence_count, window_count);
}

} // namespace rund::node::accel::detail
