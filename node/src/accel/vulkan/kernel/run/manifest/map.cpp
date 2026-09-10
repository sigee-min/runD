#include "internal.hpp"
#include "../../../../kernel/backend/source/storage.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../route.hpp"

#include "../../../kernel.hpp"
#include "../../../map/source/upper.hpp"
#include "../../pipeline/source.hpp"

#include "../../../../kernel/backend/template/source.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::node::accel::detail {

bool BuildVulkanMapManifest(const KernelExecutionStep &step,
                            const rund::kernel::ComputePlan &plan,
                            const BoundStep *const bound, const bool has_checks,
                            const bool controlled,
                            PreparedBackendManifest &manifest) noexcept {
  const std::uint64_t route_dispatches = VulkanMapRouteDispatches(plan, bound);
  const std::uint64_t checks = UniqueVulkanMapCheckCount(step.artifact);
  const std::uint64_t check_stage = has_checks ? 1u : 0u;
  const std::uint64_t control_stage = controlled ? 1u : 0u;
  manifest.source_build_count = 1u + control_stage + check_stage;
  manifest.source_library_dependency_count = manifest.source_build_count;
  manifest.pipeline_stage_count = manifest.source_build_count;
  manifest.descriptor_dependency_count = manifest.pipeline_stage_count;
  manifest.descriptor_lease_count = control_stage + check_stage;
  std::uint64_t bindings_per_window = 0u;
  std::uint64_t window_bindings = 0u;
  std::uint64_t check_bindings = 0u;
  if (!rund::kernel::checked::add(plan.input_buffer_count,
                                  plan.output_buffer_count,
                                  bindings_per_window) ||
      !rund::kernel::checked::add(bindings_per_window, 1u + control_stage,
                                  bindings_per_window) ||
      !rund::kernel::checked::mul(route_dispatches, bindings_per_window,
                                  window_bindings) ||
      !rund::kernel::checked::add(checks, 3u, check_bindings) ||
      !rund::kernel::checked::mul(check_bindings, check_stage,
                                  check_bindings) ||
      !rund::kernel::checked::add(route_dispatches, control_stage,
                                  manifest.descriptor_set_count) ||
      !rund::kernel::checked::add(manifest.descriptor_set_count, check_stage,
                                  manifest.descriptor_set_count) ||
      !rund::kernel::checked::add(window_bindings, 4u * control_stage,
                                  manifest.descriptor_binding_count) ||
      !rund::kernel::checked::add(manifest.descriptor_binding_count,
                                  check_bindings,
                                  manifest.descriptor_binding_count)) {
    return false;
  }
  std::uint64_t main_source = 0u;
  std::uint64_t final_main_source = 0u;
  std::uint64_t raw_source_transient = 0u;
  if (!backend_template_plan::map_source_upper(step, plan, main_source,
                                               raw_source_transient) ||
      (controlled && !VulkanControlledMapSourceUpperBytes(plan, main_source,
                                                          final_main_source)) ||
      !AddPreparedBackendCacheDependency(
          manifest, PreparedBackendCacheDependency{
                        .source_recipe = 0x76756c6b2e6d6170ull,
                        .source_upper_bytes =
                            controlled ? final_main_source : main_source,
                        .pipeline_stage_count = 1u,
                    })) {
    return false;
  }
  if (controlled && !backend_source_recipe::string_external_storage_upper_bytes(
                        main_source, raw_source_transient)) {
    return false;
  }
  manifest.cold_source_transient_bytes = raw_source_transient;
  if (controlled) {
    std::uint64_t control_source = 0u;
    if (!VulkanMapControlSourceBytes(control_source) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e637472ull,
                          .source_upper_bytes = control_source,
                          .pipeline_stage_count = 1u,
                      })) {
      return false;
    }
  }
  if (has_checks) {
    std::uint64_t check_source = 0u;
    if (!VulkanMapCheckSourceUpperBytes(step.artifact, check_source) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e63686bull,
                          .source_upper_bytes = check_source,
                          .pipeline_stage_count = 1u,
                      })) {
      return false;
    }
  }
  // The main Map artifact is moved into the cached pipeline. Control and
  // bounds-check collective pipelines copy their source, so their caller
  // storage is the only additional full-source allocation live during cache
  // publication. Dependency zero is always the moved main artifact.
  for (std::size_t index = 1u; index < manifest.cache_dependency_entry_count;
       ++index) {
    manifest.cold_source_transient_bytes = std::max(
        manifest.cold_source_transient_bytes,
        manifest.source_dependencies[index].source_storage_upper_bytes);
  }
  return true;
}

} // namespace rund::node::accel::detail

#endif
