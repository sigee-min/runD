#include "../../../context/internal/support.hpp"
#include "../../../kernel/backend/source/storage.hpp"
#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template/arithmetic.hpp"
#include "../../../kernel/status.hpp"
#include "../../../kernel/step/map/stride.hpp"

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
#include "map/memory.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

struct MetalRecurrenceSourceReservation final {
  std::uint64_t source_bytes{};
  std::uint64_t source_storage_bytes{};
  std::uint64_t transient_bytes{};
  std::uint64_t native_object_count{};
  bool ok{};
};

[[nodiscard]] MetalRecurrenceSourceReservation
PlanMetalRecurrenceSource(const MapRecurrenceSourcePlan &source,
                          const rund::kernel::ComputePlan &plan,
                          const bool history) noexcept {
  MetalRecurrenceSourceReservation result{};
  std::uint64_t specialized_upper = 0u;
  if (!source.ok || source.history != history ||
      source.exact_source_bytes == 0u ||
      source.source_upper_bytes < source.exact_source_bytes ||
      source.source_storage_upper_bytes == 0u ||
      !MapSpecializedSourceUpperBytes(source.exact_source_bytes,
                                      source.source_upper_bytes, plan,
                                      specialized_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(specialized_upper, 1u, true,
                                            result.source_bytes) ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          result.source_bytes, result.source_storage_bytes)) {
    return result;
  }
  // The one source allocation moves into the adapter cache and is retained
  // template storage. Only the semantic metadata that is discarded after
  // in-place specialization is transient.
  result.transient_bytes = source.metadata_storage_upper_bytes;
  // One source compilation requests one MTLLibrary and one pipeline state.
  // Their opaque byte sizes are not guessed; the structural object count is
  // the auditable native reservation.
  result.native_object_count = 2u;
  result.ok = true;
  return result;
}

} // namespace
#endif

rund::AccelCheck PlanMetalPipelineRecurrence(
    const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  if (!plan.ok) {
    return rund::AccelCheck{false, plan.reason == nullptr
                                       ? "compute_pipeline_recurrence_invalid"
                                       : plan.reason};
  }
  if (plan.group_count == 0u) {
    return plan.history_group_count == 0u
               ? rund::AccelCheck{true, "ok"}
               : rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const std::uint64_t terminal_group_count = plan.terminal_group_count();
  const bool terminal_variant = terminal_group_count != 0u;
  const bool history_variant = plan.history_group_count != 0u;
  const std::uint64_t template_count =
      static_cast<std::uint64_t>(terminal_variant) +
      static_cast<std::uint64_t>(history_variant);
  if (plan.authority == nullptr || plan.canonical_artifact == nullptr ||
      plan.canonical_artifact != &plan.authority->artifact ||
      plan.authority->kind() != rund::kernel::NodeKind::Map ||
      plan.history_group_count > plan.group_count || template_count == 0u ||
      template_count > 2u || !plan.plan.ok ||
      plan.plan.api != rund::kernel::ComputeApi::Metal ||
      plan.canonical_artifact->key.api != rund::kernel::ComputeApi::Metal ||
      plan.canonical_artifact->kind !=
          rund::kernel::LoweringArtifactKind::MetalSource ||
      (terminal_variant
           ? plan.terminal_template_group_capacity < terminal_group_count
           : plan.terminal_template_group_capacity != 0u) ||
      (history_variant
           ? plan.history_template_group_capacity < plan.history_group_count
           : plan.history_template_group_capacity != 0u) ||
      plan.window_count == 0u ||
      plan.window_count != plan.plan.dispatch_count ||
      plan.input_count != plan.plan.input_buffer_count ||
      plan.output_count != plan.plan.output_buffer_count ||
      plan.input_layouts().size() != plan.input_count ||
      plan.output_layouts().size() != plan.output_count ||
      plan.terminal_source.ok != terminal_variant ||
      plan.history_source.ok != history_variant) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (const PreparedKernelProgramBindingIdentity identity :
       plan.input_layouts()) {
    if (identity.element_bytes == 0u || identity.count == 0u ||
        identity.stride_bytes < identity.element_bytes ||
        identity.usage != rund::kernel::kResidentUsageRead) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
  }
  for (const PreparedKernelProgramBindingIdentity identity :
       plan.output_layouts()) {
    if (identity.element_bytes == 0u || identity.count == 0u ||
        identity.stride_bytes < identity.element_bytes ||
        identity.usage != rund::kernel::kResidentUsageWrite) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
  }

  std::uint64_t per_route_host = 0u;
  std::uint64_t window_bytes = 0u;
  std::uint64_t routes_host = 0u;
  std::uint64_t history_host = 0u;
  if (!AddMetalMapRouteHostBytes(per_route_host, plan.plan) ||
      !product(plan.window_count, sizeof(rund::kernel::ComputeDispatchWindow),
               window_bytes) ||
      !add(per_route_host, window_bytes) ||
      !product(per_route_host, plan.group_count, routes_host) ||
      !product(plan.history_group_count, sizeof(MapRecurrenceHistory),
               history_host) ||
      !add(routes_host, history_host) ||
      !product(plan.plan.param_bytes, plan.group_count,
               reservation.route_native_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = routes_host;

  const auto add_template = [&](const MapRecurrenceSourcePlan &source,
                                const bool history) noexcept {
    const MetalRecurrenceSourceReservation source_reservation =
        PlanMetalRecurrenceSource(source, plan.plan, history);
    std::uint64_t host = sizeof(MetalMapRecurrenceTemplate);
    if (!source_reservation.ok ||
        !AddMetalMapTemplateHostBytes(host, plan.plan, 0u) ||
        !add(host, source_reservation.source_storage_bytes) ||
        !add(reservation.template_host_bytes, host) ||
        !add(reservation.template_source_bytes,
             source_reservation.source_bytes) ||
        !add(reservation.template_native_allocation_count,
             source_reservation.native_object_count)) {
      return false;
    }
    reservation.source_transient_bytes = std::max(
        reservation.source_transient_bytes, source_reservation.transient_bytes);
    return true;
  };
  if ((terminal_variant && !add_template(plan.terminal_source, false)) ||
      (history_variant && !add_template(plan.history_source, true))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  reservation.group_count = plan.group_count;
  reservation.history_group_count = plan.history_group_count;
  reservation.template_count = template_count;
  reservation.terminal_template_group_capacity =
      plan.terminal_template_group_capacity;
  reservation.history_template_group_capacity =
      plan.history_template_group_capacity;
  reservation.route_step_count = plan.group_count;
  reservation.template_step_count = template_count;
  reservation.route_native_allocation_count = plan.group_count;
  if (reservation.group_count != plan.group_count ||
      reservation.history_group_count != plan.history_group_count ||
      reservation.template_count != template_count ||
      reservation.terminal_template_group_capacity !=
          plan.terminal_template_group_capacity ||
      reservation.history_template_group_capacity !=
          plan.history_template_group_capacity ||
      reservation.route_step_count != plan.group_count ||
      reservation.template_step_count != template_count) {
    reservation = {};
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)plan;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
