#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/source/storage.hpp"
#include "../../../kernel/backend/template/arithmetic.hpp"
#include "../../../kernel/recurrence/plan.hpp"
#include "../../../kernel/step/map/stride.hpp"
#include "../../../kernel/status.hpp"
#include "../../../resident/window/admission/runtime/windows.hpp"

#include "../../collective/chunk.hpp"
#include "../../compact/local.hpp"
#include "../../descriptor.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"
#include "../../map/source/upper.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/local/state.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/capacity.hpp"
#include "../pipeline/evidence.hpp"
#include "../pipeline/recurrence.hpp"
#include "../pipeline/source.hpp"
#include "../pipeline/state.hpp"
#include "../reset/source.hpp"

#include "../../../primitive/block.hpp"
#include "../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

#include "storage.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck PlanVulkanPipelineRecurrence(
    const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
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
      plan.plan.api != rund::kernel::ComputeApi::Vulkan ||
      plan.canonical_artifact->key.api != rund::kernel::ComputeApi::Vulkan ||
      plan.canonical_artifact->key.variant !=
          rund::kernel::LoweringArtifactVariant::Canonical ||
      plan.canonical_artifact->kind !=
          rund::kernel::LoweringArtifactKind::VulkanSource ||
      !plan.canonical_artifact->ok || !plan.canonical_artifact->metadata.ok ||
      !plan.canonical_artifact->metadata.read_routes.empty() ||
      plan.window_count == 0u ||
      plan.window_count != plan.plan.dispatch_count ||
      plan.input_count != plan.plan.input_buffer_count ||
      plan.output_count != plan.plan.output_buffer_count ||
      plan.input_layouts().size() != plan.input_count ||
      plan.output_layouts().size() != plan.output_count ||
      plan.terminal_source.ok != terminal_variant ||
      plan.history_source.ok != history_variant ||
      (terminal_variant
           ? plan.terminal_template_group_capacity < terminal_group_count
           : plan.terminal_template_group_capacity != 0u) ||
      (history_variant
           ? plan.history_template_group_capacity < plan.history_group_count
           : plan.history_template_group_capacity != 0u)) {
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

  std::uint64_t per_route_host = sizeof(VulkanMapEncodeResources);
  std::uint64_t window_bytes = 0u;
  std::uint64_t routes_host = 0u;
  std::uint64_t history_host = 0u;
  std::uint64_t per_route_native =
      std::max<std::uint64_t>(4u, plan.plan.param_bytes);
  if (!AddVulkanHostBytes(per_route_host, plan.input_count,
                          sizeof(VulkanResidentBufferResult)) ||
      !AddVulkanHostBytes(per_route_host, plan.output_count,
                          sizeof(VulkanResidentBufferResult)) ||
      !product(plan.window_count, sizeof(rund::kernel::ComputeDispatchWindow),
               window_bytes) ||
      !add(per_route_host, window_bytes) ||
      !product(per_route_host, plan.group_count, routes_host) ||
      !product(plan.history_group_count, sizeof(MapRecurrenceHistory),
               history_host) ||
      !add(routes_host, history_host) ||
      !product(per_route_native, plan.group_count,
               reservation.route_native_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = routes_host;

  std::uint64_t descriptors_per_set = 0u;
  if (!rund::kernel::checked::add(plan.input_count, plan.output_count,
                                  descriptors_per_set) ||
      !add(descriptors_per_set, 1u)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const auto add_template = [&](const MapRecurrenceSourcePlan &source,
                                const bool history,
                                const std::uint64_t group_capacity) noexcept {
    std::uint64_t expected_source_storage = 0u;
    std::uint64_t final_source_upper = 0u;
    std::uint64_t final_source_storage = 0u;
    std::uint64_t descriptor_sets = 0u;
    std::uint64_t descriptor_bindings = 0u;
    std::uint64_t host = sizeof(VulkanMapRecurrenceTemplate);
    std::uint64_t native_objects = 0u;
    if (!source.ok || source.history != history ||
        source.exact_source_bytes == 0u ||
        source.source_upper_bytes < source.exact_source_bytes ||
        !backend_source_recipe::string_external_storage_upper_bytes(
            source.source_upper_bytes, expected_source_storage) ||
        source.source_storage_upper_bytes != expected_source_storage ||
        !MapSpecializedSourceUpperBytes(source.exact_source_bytes,
                                        source.source_upper_bytes, plan.plan,
                                        final_source_upper) ||
        !backend_source_recipe::string_external_storage_upper_bytes(
            final_source_upper, final_source_storage) ||
        group_capacity == 0u ||
        !product(group_capacity, plan.window_count, descriptor_sets) ||
        descriptor_sets == 0u ||
        descriptor_sets > std::numeric_limits<std::uint32_t>::max() ||
        !add(host, sizeof(VulkanMapTemplateResources)) ||
        !add(host, sizeof(VulkanMapDescriptorArena)) ||
        !AddVulkanHostBytes(host, plan.input_count, sizeof(InputWindowPlan)) ||
        !AddVulkanHostBytes(host, plan.input_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !AddVulkanHostBytes(host, plan.output_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !add(host, sizeof(VulkanCachedPipeline)) ||
        !add(host, final_source_storage) ||
        !AddVulkanHostBytes(host, descriptor_sets, sizeof(VkDescriptorSet)) ||
        !add(reservation.template_host_bytes, host) ||
        !add(reservation.template_source_bytes, final_source_upper) ||
        !add(reservation.descriptor_set_count, descriptor_sets) ||
        !product(descriptor_sets, descriptors_per_set, descriptor_bindings) ||
        !add(reservation.descriptor_count, descriptor_bindings) ||
        !rund::kernel::checked::add(descriptor_sets, 4u, native_objects) ||
        !add(reservation.template_native_allocation_count, native_objects)) {
      return false;
    }
    // The final source allocation moves into VulkanCachedPipeline and is
    // retained template storage. Only copied semantic metadata dies after
    // the in-place specialization/cache move.
    reservation.source_transient_bytes =
        std::max(reservation.source_transient_bytes,
                 source.metadata_storage_upper_bytes);
    return true;
  };
  if ((terminal_variant &&
       !add_template(plan.terminal_source, false,
                     plan.terminal_template_group_capacity)) ||
      (history_variant &&
       !add_template(plan.history_source, true,
                     plan.history_template_group_capacity))) {
    reservation = {};
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
  return rund::AccelCheck{true, "ok"};
#else
  (void)plan;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
