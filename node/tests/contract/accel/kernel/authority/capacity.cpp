#include "src/accel/kernel/backend/pipeline/failure.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"

#include "src/accel/metal/kernel/pipeline/icb.hpp"
#include "src/accel/metal/kernel/pipeline/identity/index.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/kernel/pipeline/aggregate/source.hpp"
#include "src/accel/metal/kernel/pipeline/capture.hpp"
#include "src/accel/metal/kernel/pipeline/state.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "capacity.hpp"

namespace node_accel_contract {

[[nodiscard]] bool PreparedReservationIsFieldwiseFailClosed() {
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation limit{
      .fingerprint_hi = 0x1122334455667788ull,
      .fingerprint_lo = 0x8877665544332211ull,
      .template_step_capacity = 100u,
      .template_capacity = 100u,
      .descriptor_set_capacity = 100u,
      .descriptor_capacity = 100u,
      .ok = true,
      .reason = "ok",
  };
  constexpr std::array numeric_fields{
      &PreparedKernelPipelineReservation::host_bytes,
      &PreparedKernelPipelineReservation::native_bytes,
      &PreparedKernelPipelineReservation::route_host_bytes,
      &PreparedKernelPipelineReservation::route_native_bytes,
      &PreparedKernelPipelineReservation::template_host_bytes,
      &PreparedKernelPipelineReservation::template_native_bytes,
      &PreparedKernelPipelineReservation::template_source_bytes,
      &PreparedKernelPipelineReservation::source_transient_bytes,
      &PreparedKernelPipelineReservation::host_transient_bytes,
      &PreparedKernelPipelineReservation::route_count,
      &PreparedKernelPipelineReservation::template_count,
      &PreparedKernelPipelineReservation::template_step_count,
      &PreparedKernelPipelineReservation::route_step_count,
      &PreparedKernelPipelineReservation::authored_entry_count,
      &PreparedKernelPipelineReservation::occurrence_count,
      &PreparedKernelPipelineReservation::window_count,
      &PreparedKernelPipelineReservation::nested_group_count,
      &PreparedKernelPipelineReservation::backend_dispatch_count,
      &PreparedKernelPipelineReservation::backend_reset_dispatch_count,
      &PreparedKernelPipelineReservation::backend_window_dispatch_count,
      &PreparedKernelPipelineReservation::backend_indirect_dispatch_count,
      &PreparedKernelPipelineReservation::backend_window_state_count,
      &PreparedKernelPipelineReservation::backend_window_descriptor_state_count,
      &PreparedKernelPipelineReservation::backend_step_occurrence_count,
      &PreparedKernelPipelineReservation::backend_step_description_count,
      &PreparedKernelPipelineReservation::backend_status_source_count,
      &PreparedKernelPipelineReservation::backend_status_entry_count,
      &PreparedKernelPipelineReservation::backend_telemetry_count,
      &PreparedKernelPipelineReservation::backend_status_command_count,
      &PreparedKernelPipelineReservation::backend_telemetry_command_count,
      &PreparedKernelPipelineReservation::backend_publication_count,
      &PreparedKernelPipelineReservation::backend_terminal_publication_count,
      &PreparedKernelPipelineReservation::backend_window_control_command_count,
      &PreparedKernelPipelineReservation::backend_publication_command_count,
      &PreparedKernelPipelineReservation::backend_command_count,
      &PreparedKernelPipelineReservation::backend_command_chunk_count,
      &PreparedKernelPipelineReservation::backend_command_native_bytes,
      &PreparedKernelPipelineReservation::backend_command_binding_slot_upper,
      &PreparedKernelPipelineReservation::backend_command_binding_count,
      &PreparedKernelPipelineReservation::backend_parameter_bytes,
      &PreparedKernelPipelineReservation::backend_profile_step_count,
      &PreparedKernelPipelineReservation::backend_profile_command_count,
      &PreparedKernelPipelineReservation::backend_query_count,
      &PreparedKernelPipelineReservation::backend_native_buffer_count,
      &PreparedKernelPipelineReservation::backend_native_object_count,
      &PreparedKernelPipelineReservation::descriptor_set_count,
      &PreparedKernelPipelineReservation::descriptor_count,
      &PreparedKernelPipelineReservation::native_allocation_count,
  };
  for (const auto field : numeric_fields) {
    limit.*field = 7u;
  }
  constexpr std::array recurrence_fields{
      &PreparedMapRecurrenceReservation::route_host_bytes,
      &PreparedMapRecurrenceReservation::route_native_bytes,
      &PreparedMapRecurrenceReservation::template_host_bytes,
      &PreparedMapRecurrenceReservation::template_native_bytes,
      &PreparedMapRecurrenceReservation::template_source_bytes,
      &PreparedMapRecurrenceReservation::source_transient_bytes,
      &PreparedMapRecurrenceReservation::group_count,
      &PreparedMapRecurrenceReservation::history_group_count,
      &PreparedMapRecurrenceReservation::template_count,
      &PreparedMapRecurrenceReservation::route_step_count,
      &PreparedMapRecurrenceReservation::template_step_count,
      &PreparedMapRecurrenceReservation::descriptor_set_count,
      &PreparedMapRecurrenceReservation::descriptor_count,
      &PreparedMapRecurrenceReservation::route_native_allocation_count,
      &PreparedMapRecurrenceReservation::template_native_allocation_count,
  };
  for (const auto field : recurrence_fields) {
    limit.map_recurrence.*field = 7u;
  }
  if (!PreparedKernelPipelineReservationWithin(limit, limit)) {
    return false;
  }
  for (const auto field : numeric_fields) {
    PreparedKernelPipelineReservation candidate = limit;
    candidate.*field = limit.*field + 1u;
    if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
      return false;
    }
  }
  for (const auto field : recurrence_fields) {
    PreparedKernelPipelineReservation candidate = limit;
    candidate.map_recurrence.*field = limit.map_recurrence.*field + 1u;
    if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
      return false;
    }
  }
  for (const bool high : {true, false}) {
    PreparedKernelPipelineReservation candidate = limit;
    (high ? candidate.fingerprint_hi : candidate.fingerprint_lo) ^= 1u;
    if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
      return false;
    }
  }
  PreparedKernelPipelineReservation capacity = limit;
  capacity.template_step_capacity += 1u;
  return !PreparedKernelPipelineReservationWithin(capacity, limit);
}

[[nodiscard]] bool MetalIcbSizeClassPlanIsExactAndPortable() {
  using namespace rund::node::accel::detail;
  MetalIcbCalibration calibration{};
  for (std::size_t index = 0u; index < calibration.allocated_bytes.size();
       ++index) {
    calibration.allocated_bytes[index] =
        4096u + static_cast<std::uint64_t>(index) * 4096u;
  }
  if (!ValidMetalIcbCalibration(calibration)) {
    return false;
  }
  const MetalIcbChunkPlan empty = PlanMetalIcbChunks(0u, calibration);
  const MetalIcbChunkPlan one = PlanMetalIcbChunks(1u, calibration);
  const MetalIcbChunkPlan three = PlanMetalIcbChunks(3u, calibration);
  const MetalIcbChunkPlan full =
      PlanMetalIcbChunks(MetalPipelineIcbFullCommandCapacity, calibration);
  const MetalIcbChunkPlan next =
      PlanMetalIcbChunks(MetalPipelineIcbFullCommandCapacity + 1u, calibration);
  const MetalIcbChunkPlan multi = PlanMetalIcbChunks(
      2u * MetalPipelineIcbFullCommandCapacity + 32'769u, calibration);
  const MetalIcbChunkPlan product_actual =
      PlanMetalIcbChunks(189'825u, calibration);
  const MetalIcbChunkPlan product_upper =
      PlanMetalIcbChunks(355'766u, calibration);
  if (!empty.ok || empty.chunk_count != 0u || empty.allocated_bytes != 0u ||
      !one.ok || one.chunk_count != 1u || one.tail_command_count != 1u ||
      one.tail_command_capacity != 1u ||
      one.allocated_bytes != calibration.allocated_bytes[0u] ||
      one.retained_chunk_bytes != MetalPipelineIcbChunkHostBytes || !three.ok ||
      three.tail_command_capacity != 4u ||
      three.allocated_bytes != calibration.allocated_bytes[2u] || !full.ok ||
      full.full_chunk_count != 1u || full.tail_command_count != 0u ||
      full.chunk_count != 1u ||
      full.allocated_bytes != calibration.allocated_bytes.back() || !next.ok ||
      next.full_chunk_count != 1u || next.tail_command_count != 1u ||
      next.tail_command_capacity != 1u || next.chunk_count != 2u ||
      next.allocated_bytes != calibration.allocated_bytes.back() +
                                  calibration.allocated_bytes.front() ||
      !multi.ok || multi.full_chunk_count != 2u ||
      multi.tail_command_count != 32'769u ||
      multi.tail_command_capacity != MetalPipelineIcbFullCommandCapacity ||
      multi.chunk_count != 3u ||
      multi.allocated_bytes != 3u * calibration.allocated_bytes.back() ||
      !product_actual.ok || product_actual.chunk_count != 3u ||
      product_actual.full_chunk_count != 2u ||
      product_actual.tail_command_count != 58'753u ||
      product_actual.tail_command_capacity !=
          MetalPipelineIcbFullCommandCapacity ||
      !product_upper.ok || product_upper.chunk_count != 6u ||
      product_upper.full_chunk_count != 5u ||
      product_upper.tail_command_count != 28'086u ||
      product_upper.tail_command_capacity != 32'768u) {
    return false;
  }

  MetalIcbCalibration zero = calibration;
  zero.allocated_bytes[3u] = 0u;
  MetalIcbCalibration descending = calibration;
  descending.allocated_bytes[8u] = descending.allocated_bytes[7u] - 1u;
  MetalIcbCalibration overflow{};
  overflow.allocated_bytes.fill(std::numeric_limits<std::uint64_t>::max() / 2u +
                                1u);
  if (ValidMetalIcbCalibration(zero) || ValidMetalIcbCalibration(descending) ||
      PlanMetalIcbChunks(1u, zero).ok ||
      PlanMetalIcbChunks(3u * MetalPipelineIcbFullCommandCapacity, overflow)
          .ok ||
      MetalIcbClassIndex(0u) != MetalPipelineIcbClassCount ||
      MetalIcbClassIndex(3u) != MetalPipelineIcbClassCount ||
      MetalIcbClassIndex(MetalPipelineIcbFullCommandCapacity) !=
          MetalPipelineIcbClassCount - 1u) {
    return false;
  }

  // Physical calibration is not semantic identity. It is deliberately a
  // field-wise actual<=frozen admission dimension under the same fingerprint.
  PreparedKernelPipelineReservation limit{};
  limit.ok = true;
  limit.reason = "ok";
  limit.fingerprint_hi = 11u;
  limit.fingerprint_lo = 13u;
  limit.backend_command_chunk_count = product_upper.chunk_count;
  limit.backend_command_native_bytes = product_upper.allocated_bytes;
  PreparedKernelPipelineReservation actual = limit;
  actual.backend_command_chunk_count = product_actual.chunk_count;
  actual.backend_command_native_bytes = product_actual.allocated_bytes;
  if (!PreparedKernelPipelineReservationWithin(actual, limit)) {
    return false;
  }
  actual.backend_command_chunk_count = limit.backend_command_chunk_count + 1u;
  if (PreparedKernelPipelineReservationWithin(actual, limit)) {
    return false;
  }
  actual = limit;
  actual.backend_command_native_bytes = limit.backend_command_native_bytes + 1u;
  return !PreparedKernelPipelineReservationWithin(actual, limit);
}

[[nodiscard]] bool MetalColdIdentityIndexBudgetIsExact() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  MetalIcbCalibration calibration{};
  for (std::size_t index = 0u; index < calibration.allocated_bytes.size();
       ++index) {
    calibration.allocated_bytes[index] =
        4096u + static_cast<std::uint64_t>(index) * 4096u;
  }
  const auto plan = [&](PreparedKernelPipelineReservation &value) {
    return PlanMetalPipelineStructureForCalibration(calibration, value);
  };
  PreparedKernelPipelineReservation reservation{};
  reservation.backend_dispatch_count = 1u;
  const rund::AccelCheck planned = plan(reservation);
  const MetalPointerIdentityIndexLayout index = PlanMetalPointerIdentityIndex(
      std::max(reservation.backend_command_count,
               reservation.backend_command_binding_count));
  if (!planned.ok || !index.ok || reservation.backend_command_count != 3u ||
      reservation.backend_command_binding_slot_upper != 4u ||
      reservation.backend_command_binding_count != 15u ||
      reservation.backend_parameter_bytes !=
          3u * sizeof(MetalNestedAggregateParams) ||
      reservation.backend_command_chunk_count != 1u ||
      reservation.native_bytes !=
          reservation.backend_parameter_bytes +
              reservation.backend_command_native_bytes ||
      reservation.host_transient_bytes !=
          index.byte_count + reservation.backend_parameter_bytes ||
      reservation.host_transient_bytes == 0u) {
    return false;
  }
  PreparedKernelPipelineReservation windows{};
  windows.backend_dispatch_count = 1u;
  windows.backend_window_dispatch_count = 7u;
  windows.window_count = 5u;
  const rund::AccelCheck windows_planned = plan(windows);
  const MetalPointerIdentityIndexLayout windows_index =
      PlanMetalPointerIdentityIndex(
          std::max(windows.backend_command_count,
                   windows.backend_command_binding_count));
  if (!windows_planned.ok || !windows_index.ok ||
      windows.backend_command_count != 8u ||
      windows.backend_window_control_command_count != 5u ||
      windows.backend_command_binding_slot_upper != 8u ||
      windows.backend_command_binding_count !=
          windows.backend_command_count * 9u ||
      windows.backend_command_chunk_count != 1u ||
      windows.native_bytes != windows.backend_parameter_bytes +
                                  windows.backend_command_native_bytes ||
      windows.host_transient_bytes != windows_index.byte_count +
                                          5u * sizeof(MetalWindow) +
                                          windows.backend_parameter_bytes) {
    return false;
  }
  PreparedKernelPipelineReservation telemetry{};
  telemetry.backend_dispatch_count = 1u;
  telemetry.backend_telemetry_command_count = 1u;
  PreparedKernelPipelineReservation status{};
  status.backend_dispatch_count = 1u;
  status.backend_status_source_count = 1u;
  status.backend_status_command_count = 1u;
  status.backend_profile_step_count = 1u;
  PreparedKernelPipelineReservation route{};
  route.backend_dispatch_count = 1u;
  route.backend_command_binding_slot_upper = 11u;
  PreparedKernelPipelineReservation publication{};
  publication.backend_dispatch_count = 1u;
  publication.backend_publication_count = 99u;
  publication.backend_publication_command_count = 3u;
  return plan(telemetry).ok &&
         telemetry.backend_command_binding_slot_upper == 9u &&
         telemetry.backend_command_binding_count ==
             telemetry.backend_command_count * 10u &&
         plan(status).ok && status.backend_command_binding_slot_upper == 7u &&
         status.backend_command_binding_count ==
             status.backend_command_count * 8u &&
         plan(route).ok && route.backend_command_binding_slot_upper == 11u &&
         route.backend_command_binding_count ==
             route.backend_command_count * 12u &&
         plan(publication).ok && publication.backend_command_count == 6u &&
         publication.backend_command_binding_slot_upper == 8u;
#else
  return true;
#endif
}

[[nodiscard]] bool MetalCaptureRowCapacityIsExact() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using rund::node::accel::detail::PlanMetalCaptureRowCapacity;
  using rund::node::accel::detail::PlanMetalParameterCapacity;
  const auto empty = PlanMetalCaptureRowCapacity(0u);
  const auto one = PlanMetalCaptureRowCapacity(1u);
  const auto eight = PlanMetalCaptureRowCapacity(8u);
  const auto nine = PlanMetalCaptureRowCapacity(9u);
  const auto overflow =
      PlanMetalCaptureRowCapacity(std::numeric_limits<std::uint64_t>::max());
  const auto parameter_first = PlanMetalParameterCapacity(0u, 1u, 1000u);
  const auto parameter_second = PlanMetalParameterCapacity(256u, 257u, 1000u);
  const auto parameter_capped = PlanMetalParameterCapacity(512u, 999u, 1000u);
  const auto parameter_stable = PlanMetalParameterCapacity(1000u, 1000u, 1000u);
  const auto parameter_overflow =
      PlanMetalParameterCapacity(1000u, 1001u, 1000u);
  return empty.ok && empty.rows == 0u && one.ok && one.rows == 1u && eight.ok &&
         eight.rows == 8u && nine.ok && nine.rows == 9u && !overflow.ok &&
         parameter_first.ok && parameter_first.rows == 256u &&
         parameter_second.ok && parameter_second.rows == 512u &&
         parameter_capped.ok && parameter_capped.rows == 1000u &&
         parameter_stable.ok && parameter_stable.rows == 1000u &&
         !parameter_overflow.ok;
#else
  return true;
#endif
}

[[nodiscard]] bool PreparedTemplateStepCapacityIsBackendBounded() {
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation limit{
      .fingerprint_hi = 0x74656d706c617465ull,
      .fingerprint_lo = 0x737465702e636170ull,
      // Plan-only inspection may freeze demand above the backend's native
      // materialization ceiling; the runtime candidate must still fit it.
      .template_step_count = 9u,
      .template_step_capacity = 8u,
      .ok = true,
      .reason = "ok",
  };
  PreparedKernelPipelineReservation candidate = limit;
  candidate.template_step_count = 8u;
  if (!PreparedKernelPipelineReservationWithin(candidate, limit)) {
    return false;
  }
  candidate.template_step_count = 9u;
  if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
    return false;
  }
  candidate.template_step_count = 8u;
  candidate.template_step_capacity = 9u;
  if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
    return false;
  }

  PreparedPipelineFailureContext failure{};
  failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  failure.template_node_route(7u, 11u);
  const PreparedPipelineFailure exact =
      failure.failure(PreparedPipelineTemplateStepCapacityReasonKey);
  return exact.stage == PreparedPipelineFailureStage::CommonAccounting &&
         exact.template_index == 7u &&
         exact.occurrence_index == PreparedPipelineUnknownCoordinate &&
         exact.node == 11u &&
         exact.outer_iteration == PreparedPipelineUnknownCoordinate &&
         exact.inner_iteration == PreparedPipelineUnknownCoordinate &&
         exact.nested_phase == rund::compute::PipelineNestedPhase::None &&
         std::string_view{exact.native_reason_key} ==
             PreparedPipelineTemplateStepCapacityReasonKey;
}

} // namespace node_accel_contract
