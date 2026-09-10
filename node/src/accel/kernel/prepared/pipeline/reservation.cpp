#include "../template/registry/reservation.hpp"
#include "reservation.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::node::accel::detail {

bool accumulate(std::uint64_t &target,
                const std::uint64_t value) noexcept {
  return ::rund::kernel::checked::add(target, value, target);
}

bool accumulate_reservation(
    PreparedKernelPipelineReservation &target,
    const PreparedKernelPipelineReservation &value) noexcept {
  target.source_transient_bytes =
      std::max(target.source_transient_bytes, value.source_transient_bytes);
  target.host_transient_bytes =
      std::max(target.host_transient_bytes, value.host_transient_bytes);
  target.backend_command_binding_slot_upper =
      std::max(target.backend_command_binding_slot_upper,
               value.backend_command_binding_slot_upper);
  return accumulate(target.host_bytes, value.host_bytes) &&
         accumulate(target.native_bytes, value.native_bytes) &&
         accumulate(target.route_host_bytes, value.route_host_bytes) &&
         accumulate(target.route_native_bytes, value.route_native_bytes) &&
         accumulate(target.template_host_bytes, value.template_host_bytes) &&
         accumulate(target.template_native_bytes,
                    value.template_native_bytes) &&
         accumulate(target.template_source_bytes,
                    value.template_source_bytes) &&
         accumulate(target.route_count, value.route_count) &&
         accumulate(target.template_count, value.template_count) &&
         accumulate(target.template_step_count, value.template_step_count) &&
         accumulate(target.route_step_count, value.route_step_count) &&
         accumulate(target.authored_entry_count, value.authored_entry_count) &&
         accumulate(target.occurrence_count, value.occurrence_count) &&
         accumulate(target.window_count, value.window_count) &&
         accumulate(target.nested_group_count, value.nested_group_count) &&
         accumulate(target.backend_dispatch_count,
                    value.backend_dispatch_count) &&
         accumulate(target.backend_reset_dispatch_count,
                    value.backend_reset_dispatch_count) &&
         accumulate(target.backend_window_dispatch_count,
                    value.backend_window_dispatch_count) &&
         accumulate(target.backend_indirect_dispatch_count,
                    value.backend_indirect_dispatch_count) &&
         accumulate(target.backend_window_state_count,
                    value.backend_window_state_count) &&
         accumulate(target.backend_window_descriptor_state_count,
                    value.backend_window_descriptor_state_count) &&
         accumulate(target.backend_step_occurrence_count,
                    value.backend_step_occurrence_count) &&
         accumulate(target.backend_step_description_count,
                    value.backend_step_description_count) &&
         accumulate(target.backend_status_source_count,
                    value.backend_status_source_count) &&
         accumulate(target.backend_status_entry_count,
                    value.backend_status_entry_count) &&
         accumulate(target.backend_telemetry_count,
                    value.backend_telemetry_count) &&
         accumulate(target.backend_status_command_count,
                    value.backend_status_command_count) &&
         accumulate(target.backend_telemetry_command_count,
                    value.backend_telemetry_command_count) &&
         accumulate(target.backend_publication_count,
                    value.backend_publication_count) &&
         accumulate(target.backend_terminal_publication_count,
                    value.backend_terminal_publication_count) &&
         accumulate(target.backend_window_control_command_count,
                    value.backend_window_control_command_count) &&
         accumulate(target.backend_publication_command_count,
                    value.backend_publication_command_count) &&
         accumulate(target.backend_command_count, value.backend_command_count) &&
         accumulate(target.backend_command_chunk_count,
                    value.backend_command_chunk_count) &&
         accumulate(target.backend_command_native_bytes,
                    value.backend_command_native_bytes) &&
         accumulate(target.backend_command_binding_count,
                    value.backend_command_binding_count) &&
         accumulate(target.backend_parameter_bytes,
                    value.backend_parameter_bytes) &&
         accumulate(target.backend_profile_step_count,
                    value.backend_profile_step_count) &&
         accumulate(target.backend_profile_command_count,
                    value.backend_profile_command_count) &&
         accumulate(target.backend_query_count, value.backend_query_count) &&
         accumulate(target.backend_native_buffer_count,
                    value.backend_native_buffer_count) &&
         accumulate(target.backend_native_object_count,
                    value.backend_native_object_count) &&
         accumulate(target.descriptor_set_count, value.descriptor_set_count) &&
         accumulate(target.descriptor_count, value.descriptor_count) &&
         accumulate(target.native_allocation_count,
                    value.native_allocation_count) &&
         accumulate(target.service_free_direct_host_bytes,
                    value.service_free_direct_host_bytes) &&
         accumulate(target.map_recurrence.route_host_bytes,
                    value.map_recurrence.route_host_bytes) &&
         accumulate(target.map_recurrence.route_native_bytes,
                    value.map_recurrence.route_native_bytes) &&
         accumulate(target.map_recurrence.template_host_bytes,
                    value.map_recurrence.template_host_bytes) &&
         accumulate(target.map_recurrence.template_native_bytes,
                    value.map_recurrence.template_native_bytes) &&
         accumulate(target.map_recurrence.template_source_bytes,
                    value.map_recurrence.template_source_bytes) &&
         accumulate(target.map_recurrence.group_count,
                    value.map_recurrence.group_count) &&
         accumulate(target.map_recurrence.history_group_count,
                    value.map_recurrence.history_group_count) &&
         accumulate(target.map_recurrence.template_count,
                    value.map_recurrence.template_count) &&
         accumulate(target.map_recurrence.terminal_template_group_capacity,
                    value.map_recurrence.terminal_template_group_capacity) &&
         accumulate(target.map_recurrence.history_template_group_capacity,
                    value.map_recurrence.history_template_group_capacity) &&
         accumulate(target.map_recurrence.route_step_count,
                    value.map_recurrence.route_step_count) &&
         accumulate(target.map_recurrence.template_step_count,
                    value.map_recurrence.template_step_count) &&
         accumulate(target.map_recurrence.descriptor_set_count,
                    value.map_recurrence.descriptor_set_count) &&
         accumulate(target.map_recurrence.descriptor_count,
                    value.map_recurrence.descriptor_count) &&
         accumulate(target.map_recurrence.route_native_allocation_count,
                    value.map_recurrence.route_native_allocation_count) &&
         accumulate(target.map_recurrence.template_native_allocation_count,
                    value.map_recurrence.template_native_allocation_count) &&
         (target.map_recurrence.source_transient_bytes =
              std::max(target.map_recurrence.source_transient_bytes,
                       value.map_recurrence.source_transient_bytes),
          true);
}

bool PreparedKernelPipelineReservationWithin(
    const PreparedKernelPipelineReservation &reservation,
    const PreparedKernelPipelineReservation &limit) noexcept {
  return reservation.ok && limit.ok &&
         reservation.fingerprint_hi == limit.fingerprint_hi &&
         reservation.fingerprint_lo == limit.fingerprint_lo &&
         reservation.host_bytes <= limit.host_bytes &&
         reservation.native_bytes <= limit.native_bytes &&
         reservation.route_host_bytes <= limit.route_host_bytes &&
         reservation.route_native_bytes <= limit.route_native_bytes &&
         reservation.template_host_bytes <= limit.template_host_bytes &&
         reservation.template_native_bytes <= limit.template_native_bytes &&
         reservation.template_source_bytes <= limit.template_source_bytes &&
         reservation.source_transient_bytes <= limit.source_transient_bytes &&
         reservation.host_transient_bytes <= limit.host_transient_bytes &&
         reservation.route_count <= limit.route_count &&
         reservation.template_count <= limit.template_count &&
         reservation.template_step_count <= limit.template_step_count &&
         reservation.template_step_count <= limit.template_step_capacity &&
         reservation.template_step_capacity == limit.template_step_capacity &&
         reservation.route_step_count <= limit.route_step_count &&
         reservation.authored_entry_count <= limit.authored_entry_count &&
         reservation.occurrence_count <= limit.occurrence_count &&
         reservation.window_count <= limit.window_count &&
         reservation.nested_group_count <= limit.nested_group_count &&
         reservation.backend_dispatch_count <= limit.backend_dispatch_count &&
         reservation.backend_reset_dispatch_count <=
             limit.backend_reset_dispatch_count &&
         reservation.backend_window_dispatch_count <=
             limit.backend_window_dispatch_count &&
         reservation.backend_indirect_dispatch_count <=
             limit.backend_indirect_dispatch_count &&
         reservation.backend_window_state_count <=
             limit.backend_window_state_count &&
         reservation.backend_window_descriptor_state_count <=
             limit.backend_window_descriptor_state_count &&
         reservation.backend_step_occurrence_count <=
             limit.backend_step_occurrence_count &&
         reservation.backend_step_description_count <=
             limit.backend_step_description_count &&
         reservation.backend_status_source_count <=
             limit.backend_status_source_count &&
         reservation.backend_status_entry_count <=
             limit.backend_status_entry_count &&
         reservation.backend_telemetry_count <= limit.backend_telemetry_count &&
         reservation.backend_status_command_count <=
             limit.backend_status_command_count &&
         reservation.backend_telemetry_command_count <=
             limit.backend_telemetry_command_count &&
         reservation.backend_publication_count <=
             limit.backend_publication_count &&
         reservation.backend_terminal_publication_count <=
             limit.backend_terminal_publication_count &&
         reservation.backend_window_control_command_count <=
             limit.backend_window_control_command_count &&
         reservation.backend_publication_command_count <=
             limit.backend_publication_command_count &&
         reservation.backend_command_count <= limit.backend_command_count &&
         reservation.backend_command_chunk_count <=
             limit.backend_command_chunk_count &&
         reservation.backend_command_native_bytes <=
             limit.backend_command_native_bytes &&
         reservation.backend_command_binding_slot_upper <=
             limit.backend_command_binding_slot_upper &&
         reservation.backend_command_binding_count <=
             limit.backend_command_binding_count &&
         reservation.backend_parameter_bytes <= limit.backend_parameter_bytes &&
         reservation.backend_profile_step_count <=
             limit.backend_profile_step_count &&
         reservation.backend_profile_command_count <=
             limit.backend_profile_command_count &&
         reservation.backend_query_count <= limit.backend_query_count &&
         reservation.backend_native_buffer_count <=
             limit.backend_native_buffer_count &&
         reservation.backend_native_object_count <=
             limit.backend_native_object_count &&
         reservation.descriptor_set_count <= limit.descriptor_set_count &&
         reservation.descriptor_count <= limit.descriptor_count &&
         reservation.native_allocation_count <= limit.native_allocation_count &&
         reservation.map_recurrence.route_host_bytes <=
             limit.map_recurrence.route_host_bytes &&
         reservation.map_recurrence.route_native_bytes <=
             limit.map_recurrence.route_native_bytes &&
         reservation.map_recurrence.template_host_bytes <=
             limit.map_recurrence.template_host_bytes &&
         reservation.map_recurrence.template_native_bytes <=
             limit.map_recurrence.template_native_bytes &&
         reservation.map_recurrence.template_source_bytes <=
             limit.map_recurrence.template_source_bytes &&
         reservation.map_recurrence.source_transient_bytes <=
             limit.map_recurrence.source_transient_bytes &&
         reservation.map_recurrence.group_count <=
             limit.map_recurrence.group_count &&
         reservation.map_recurrence.history_group_count <=
             limit.map_recurrence.history_group_count &&
         reservation.map_recurrence.template_count <=
             limit.map_recurrence.template_count &&
         reservation.map_recurrence.terminal_template_group_capacity <=
             limit.map_recurrence.terminal_template_group_capacity &&
         reservation.map_recurrence.history_template_group_capacity <=
             limit.map_recurrence.history_template_group_capacity &&
         reservation.map_recurrence.route_step_count <=
             limit.map_recurrence.route_step_count &&
         reservation.map_recurrence.template_step_count <=
             limit.map_recurrence.template_step_count &&
         reservation.map_recurrence.descriptor_set_count <=
             limit.map_recurrence.descriptor_set_count &&
         reservation.map_recurrence.descriptor_count <=
             limit.map_recurrence.descriptor_count &&
         reservation.map_recurrence.route_native_allocation_count <=
             limit.map_recurrence.route_native_allocation_count &&
         reservation.map_recurrence.template_native_allocation_count <=
             limit.map_recurrence.template_native_allocation_count &&
         reservation.service_free_direct_host_bytes <=
             limit.service_free_direct_host_bytes &&
         reservation.template_count <= limit.template_capacity &&
         reservation.descriptor_set_count <= limit.descriptor_set_capacity &&
         reservation.descriptor_count <= limit.descriptor_capacity;
}

} // namespace rund::node::accel::detail
