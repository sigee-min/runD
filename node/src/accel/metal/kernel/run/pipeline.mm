#include "../../../context/internal/support.hpp"
#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template/arithmetic.hpp"
#include "../../../kernel/status.hpp"

#include "../../../sort/block/metal.hpp"
#include "../../buffer/owner.hpp"
#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../pipeline/guard.hpp"
#include "../../pipeline/source/recipe.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../runtime/map/source/upper.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/source.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/build.hpp"
#include "../pipeline/identity/index.hpp"
#include "parameter.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck PlanMetalPipelineStructureForCalibration(
    const MetalIcbCalibration &calibration,
    PreparedKernelPipelineReservation &reservation) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  std::uint64_t commands = reservation.backend_dispatch_count;
  const std::uint64_t status_reset_command_count =
      reservation.backend_status_source_count == 0u ? 0u : 1u;
  for (const std::uint64_t count :
       {reservation.backend_reset_dispatch_count,
        reservation.backend_status_command_count,
        reservation.backend_telemetry_command_count, reservation.window_count,
        reservation.backend_publication_command_count,
        status_reset_command_count, std::uint64_t{2u}}) {
    if (!add(commands, count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  std::uint64_t bindings = 0u;
  std::uint64_t binding_slot_upper =
      std::max(reservation.backend_command_binding_slot_upper,
               std::uint64_t{4u}); // pipeline open/close: indices 0...3
  if (reservation.backend_reset_dispatch_count != 0u) {
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{2u});
  }
  if (status_reset_command_count != 0u) {
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{3u});
  }
  if (reservation.backend_status_command_count != 0u) {
    // Status fold binds 0...5, plus step-control at 6 under profiling.
    binding_slot_upper = std::max(binding_slot_upper,
                                  reservation.backend_profile_step_count == 0u
                                      ? std::uint64_t{6u}
                                      : std::uint64_t{7u});
  }
  if (reservation.backend_telemetry_command_count != 0u) {
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{9u});
  }
  if (reservation.window_count != 0u ||
      reservation.backend_publication_command_count != 0u ||
      reservation.nested_group_count != 0u) {
    // Window canonicalization/publication and the nested aggregate author
    // indices 0...7. Advance alone is a strict subset.
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{8u});
  }
  std::uint64_t captured_binding_slots = binding_slot_upper;
  if (binding_slot_upper > kMetalPipelineGuardBinding ||
      !add(captured_binding_slots, 1u)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t parameter_bytes = reservation.backend_parameter_bytes;
  const MetalIcbChunkPlan icb_plan = PlanMetalIcbChunks(commands, calibration);
  if (!icb_plan.ok || icb_plan.chunk_count == 0u ||
      icb_plan.allocated_bytes == 0u) {
    return rund::AccelCheck{false, "accel_metal_icb_calibration_failed"};
  }
  std::uint64_t host = sizeof(MetalSequence);
  std::uint64_t bytes = 0u;
  MetalCaptureRowCapacity command_rows{};
  MetalCaptureRowCapacity binding_rows{};
  // Every admitted spatial local contributes physical execution steps. This
  // existing occurrence bound covers its cold-sized proof rows without
  // retaining the global step capacity in every ordinary sequence.
  if (!product(reservation.backend_step_occurrence_count,
               sizeof(MetalSpatialWindowLocalProof), bytes) ||
      !add(host, bytes) || !add(host, icb_plan.retained_chunk_bytes) ||
      !product(commands, captured_binding_slots, bindings) ||
      !(command_rows = PlanMetalCaptureRowCapacity(commands)).ok ||
      !(binding_rows = PlanMetalCaptureRowCapacity(bindings)).ok ||
      !product(command_rows.rows, sizeof(MetalCommand), bytes) ||
      !add(host, bytes) ||
      !product(binding_rows.rows, sizeof(MetalCommandBinding), bytes) ||
      !add(host, bytes) ||
      !product(commands, sizeof(id<MTLComputePipelineState>), bytes) ||
      !add(host, bytes) || !product(bindings, sizeof(id<MTLResource>), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_step_description_count,
               sizeof(MetalPipelineTelemetryRecord), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_profile_step_count,
               sizeof(PreparedPipelineStepEvidence), bytes) ||
      !add(host, bytes) ||
      !product(reservation.nested_group_count, sizeof(std::shared_ptr<void>),
               bytes) ||
      !add(host, bytes) || !add(reservation.host_bytes, host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Fixed inline payload is bounded per command, not per argument slot. Most
  // encoders bind one params record. The only multi-setBytes commands are Scan
  // (at most six aligned scalars), telemetry (params + two scalars), status
  // reset, and status fold. Their totals fit this row; variable status/reset
  // arrays are occurrence-weighted separately above. Multiplying by all 31
  // argument slots manufactured nearly 2 GiB of unreachable parameter memory
  // in the large recurrence product case.
  constexpr std::uint64_t FixedParameterUpper =
      sizeof(MetalNestedAggregateParams);
  static_assert(FixedParameterUpper == 176u);
  static_assert(FixedParameterUpper >= sizeof(MetalPublishParams));
  static_assert(FixedParameterUpper >= sizeof(MetalWindowParams));
  static_assert(FixedParameterUpper >= sizeof(MetalPipelineTelemetryParams));
  static_assert(FixedParameterUpper >= sizeof(MetalPipelineStatusParams));
  static_assert(FixedParameterUpper >= sizeof(NumericParams));
  static_assert(FixedParameterUpper >= sizeof(reset::Params));
  static_assert(FixedParameterUpper >= 6u * 16u);
  static_assert(FixedParameterUpper >= 80u + 16u + 16u);
  std::uint64_t fixed_parameter_bytes = 0u;
  if (!product(commands, FixedParameterUpper, fixed_parameter_bytes) ||
      !add(parameter_bytes, fixed_parameter_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (status_reset_command_count != 0u) {
    std::uint64_t reset_rows = 0u;
    std::uint64_t aligned = 0u;
    if (!product(reservation.backend_status_source_count,
                 sizeof(MetalPipelineResetMeta), reset_rows) ||
        !AddAlignedMetalParameterBytes(aligned, reset_rows) ||
        !add(parameter_bytes, aligned)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!add(reservation.native_bytes, parameter_bytes) ||
      !add(reservation.native_bytes, icb_plan.allocated_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const MetalPointerIdentityIndexLayout identity_index =
      PlanMetalPointerIdentityIndex(std::max(commands, bindings));
  if (!identity_index.ok) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t native_window_bytes = 0u;
  std::uint64_t finalizer_transient_bytes = identity_index.byte_count;
  std::uint64_t status_binding_capacity = 0u;
  std::uint64_t status_description_bytes = 0u;
  if (!product(reservation.window_count, sizeof(MetalWindow),
               native_window_bytes) ||
      !add(finalizer_transient_bytes, native_window_bytes) ||
      !product(reservation.backend_step_description_count,
               kMetalPipelineStatusBindingCapacity, status_binding_capacity) ||
      !product(status_binding_capacity,
               sizeof(MetalPipelineStatusBindingRecord) +
                   sizeof(MetalPipelineStatusSourceMeta) +
                   sizeof(MetalPipelineResetMeta),
               status_description_bytes) ||
      !add(finalizer_transient_bytes, status_description_bytes) ||
      !product(reservation.backend_status_entry_count,
               sizeof(MetalPipelineStatusEntryMeta),
               status_description_bytes) ||
      !add(finalizer_transient_bytes, status_description_bytes) ||
      !product(reservation.backend_step_description_count,
               sizeof(PreparedProgramStatusSlice), status_description_bytes) ||
      !add(finalizer_transient_bytes, status_description_bytes) ||
      !add(finalizer_transient_bytes, parameter_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Resolved native window rows remain live until ICB finalization and
  // therefore coexist with the pointer-identity workspace, status-description
  // rows, and captured parameter bytes. Occurrence counts are conservative
  // frozen uppers for template-local description rows; transducer fusion may
  // use less, but no unplanned heap is hidden below this boundary.
  reservation.host_transient_bytes =
      std::max(reservation.host_transient_bytes, finalizer_transient_bytes);
  reservation.backend_command_count = commands;
  reservation.backend_command_chunk_count = icb_plan.chunk_count;
  reservation.backend_command_native_bytes = icb_plan.allocated_bytes;
  reservation.backend_command_binding_slot_upper = binding_slot_upper;
  reservation.backend_command_binding_count = bindings;
  reservation.backend_parameter_bytes = parameter_bytes;
  reservation.backend_window_control_command_count = reservation.window_count;
  // Retained command stream: the exact size-class ICB chunk count plus the
  // common control/status buffers; optional parameter, resident-state, and
  // profile buffers are explicit.
  reservation.backend_native_buffer_count =
      3u + static_cast<std::uint64_t>(parameter_bytes != 0u) +
      static_cast<std::uint64_t>(reservation.window_count != 0u) +
      static_cast<std::uint64_t>(reservation.backend_profile_step_count != 0u);
  reservation.backend_native_object_count =
      icb_plan.chunk_count + reservation.backend_native_buffer_count;
  if (!add(reservation.native_allocation_count,
           reservation.backend_native_object_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)calibration;
  (void)reservation;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck PlanMetalPipelineStructure(
    const rund::AccelContext &context,
    PreparedKernelPipelineReservation &reservation) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const ContextAdmission admission = AdmitContextForSupport(context);
  MetalAdapter *const adapter = admission.pick == nullptr
                                    ? nullptr
                                    : MetalAdapterFromPick(admission.pick->raw);
  if (!admission.check.ok || admission.api != rund::AccelApi::Metal ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (!ValidMetalIcbCalibration(adapter->pipeline_icb_calibration)) {
    return rund::AccelCheck{false, "accel_metal_icb_calibration_failed"};
  }
  return PlanMetalPipelineStructureForCalibration(
      adapter->pipeline_icb_calibration, reservation);
#else
  (void)context;
  (void)reservation;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
