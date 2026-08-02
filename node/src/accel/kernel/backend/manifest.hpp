#pragma once

#include "source_recipe.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

// One step can own at most five distinct source-library dependencies today:
// Vulkan Sort materializes five stage sources.  This is a semantic bound from
// the exhaustive primitive table, not a route/template count limit.
inline constexpr std::size_t PreparedBackendCacheDependencyCapacity = 5u;

struct PreparedBackendCacheDependency final {
  // Diagnostic recipe discriminator only. Entries are deliberately not
  // deduplicated: a hash-sized projection cannot be collision-safe authority
  // for a global adapter cache key. Cross-step grouping remains disabled until
  // a builder-adjacent complete cache tuple is available.
  std::uint64_t source_recipe{};
  std::uint64_t source_upper_bytes{};
  std::uint64_t pipeline_stage_count{};
  // Conservative std::string external-storage envelope for the retained
  // source. Text cardinality remains independently exact above.
  std::uint64_t source_storage_upper_bytes{};

  [[nodiscard]] constexpr bool complete() const noexcept {
    return source_recipe != 0u && source_upper_bytes != 0u &&
           pipeline_stage_count != 0u && source_storage_upper_bytes != 0u;
  }
};

// Allocation-free execution-owner projection shared by cold planning and
// backend preparation. Counts describe semantic work requested from an empty
// adapter cache; cache-hit/miss growth remains runtime telemetry.
struct PreparedBackendManifest final {
  std::uint64_t source_build_count{};
  // U: number of retained source/library cache dependencies. This is not P:
  // one source library may produce several pipeline stages (Metal Sort).
  std::uint64_t source_library_dependency_count{};
  std::uint64_t pipeline_stage_count{};
  std::uint64_t descriptor_set_count{};
  std::uint64_t descriptor_binding_count{};
  std::uint64_t descriptor_lease_count{};
  std::uint64_t descriptor_dependency_count{};
  // Exact encoder command cardinality for one prepared route occurrence.
  // Direct and device-authored indirect dispatches are deliberately separate:
  // both consume one window capture slot, while only indirect dispatches
  // consume a gate descriptor.
  std::uint64_t capture_direct_dispatch_count{};
  std::uint64_t capture_indirect_dispatch_count{};
  // Highest non-guard argument index authored by this producer, plus one.
  // Metal snapshots retain the encoder binding mask between commands, so the
  // pipeline planner combines this prefix with other producers by max rather
  // than summing unrelated slot spaces.
  std::uint64_t capture_binding_slot_upper{};
  // Exact canonical control sources described by this step. These are not
  // semantic operation/pass counts and never stand in for capture commands.
  std::uint64_t status_source_count{};
  // Exact packed status entries owned by those sources. Element cardinality
  // is not a valid proxy: many primitives expose one aggregate status while
  // numeric batches may expose one status per batch.
  std::uint64_t status_entry_count{};
  // Exact status commands and aligned setBytes/push-constant payload emitted
  // by one physical occurrence. Source count alone is insufficient: one
  // source may require import plus reduction, while several sources may share
  // one reduction command.
  std::uint64_t status_command_count{};
  std::uint64_t status_parameter_bytes{};
  std::uint64_t telemetry_source_count{};
  // Exact authored UTF-8 source bytes retained by cache dependencies.
  std::uint64_t cold_cache_source_bytes{};
  // Conservative external std::string storage for those same sources.
  std::uint64_t cold_cache_source_storage_bytes{};
  // Largest raw source allocation that can coexist while a Pipeline-private
  // wrapper grows one retained source in place.
  std::uint64_t cold_source_transient_bytes{};
  std::uint64_t cold_cache_native_object_count{};
  std::uint64_t cache_dependency_entry_count{};
  std::array<PreparedBackendCacheDependency,
             PreparedBackendCacheDependencyCapacity>
      source_dependencies{};
  bool source_dependencies_complete{};
  bool ok{};
  const char *reason{"compute_pipeline_capacity"};
};

[[nodiscard]] inline bool ValidPreparedBackendControlManifest(
    const PreparedBackendManifest &manifest) noexcept {
  const bool has_status = manifest.status_source_count != 0u;
  return has_status == (manifest.status_entry_count != 0u) &&
         has_status == (manifest.status_command_count != 0u) &&
         has_status == (manifest.status_parameter_bytes != 0u);
}

[[nodiscard]] inline bool AddPreparedBackendCacheDependency(
    PreparedBackendManifest &manifest,
    PreparedBackendCacheDependency dependency) noexcept {
  if (dependency.source_recipe == 0u || dependency.source_upper_bytes == 0u ||
      dependency.pipeline_stage_count == 0u ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          dependency.source_upper_bytes,
          dependency.source_storage_upper_bytes) ||
      !dependency.complete()) {
    manifest.source_dependencies_complete = false;
    return false;
  }
  if (manifest.cache_dependency_entry_count ==
          manifest.source_dependencies.size() ||
      !rund::kernel::checked::add(manifest.cold_cache_source_bytes,
                                  dependency.source_upper_bytes,
                                  manifest.cold_cache_source_bytes) ||
      !rund::kernel::checked::add(manifest.cold_cache_source_storage_bytes,
                                  dependency.source_storage_upper_bytes,
                                  manifest.cold_cache_source_storage_bytes)) {
    manifest.source_dependencies_complete = false;
    return false;
  }
  manifest.source_dependencies[manifest.cache_dependency_entry_count++] =
      dependency;
  return true;
}

[[nodiscard]] inline bool
CompleteVulkanBackendManifest(PreparedBackendManifest &manifest) noexcept {
  std::uint64_t pipeline_objects = 0u;
  std::uint64_t descriptor_objects = 0u;
  std::uint64_t capture_dispatches = 0u;
  bool dependencies_complete = manifest.source_library_dependency_count != 0u &&
                               manifest.cache_dependency_entry_count ==
                                   manifest.source_library_dependency_count;
  for (std::size_t index = 0u; index < manifest.cache_dependency_entry_count;
       ++index) {
    dependencies_complete =
        dependencies_complete && manifest.source_dependencies[index].complete();
  }
  manifest.source_dependencies_complete = dependencies_complete;
  if (manifest.pipeline_stage_count == 0u ||
      manifest.source_library_dependency_count == 0u ||
      !ValidPreparedBackendControlManifest(manifest) ||
      !manifest.source_dependencies_complete ||
      !rund::kernel::checked::add(manifest.capture_direct_dispatch_count,
                                  manifest.capture_indirect_dispatch_count,
                                  capture_dispatches) ||
      capture_dispatches == 0u ||
      !rund::kernel::checked::mul(manifest.pipeline_stage_count, 3u,
                                  pipeline_objects) ||
      !rund::kernel::checked::add(manifest.descriptor_set_count,
                                  manifest.descriptor_dependency_count,
                                  descriptor_objects) ||
      !rund::kernel::checked::add(pipeline_objects, descriptor_objects,
                                  manifest.cold_cache_native_object_count)) {
    manifest.ok = false;
    manifest.reason = "compute_pipeline_capacity";
    return false;
  }
  manifest.ok = true;
  manifest.reason = "ok";
  return true;
}

[[nodiscard]] inline bool
CompleteMetalBackendManifest(PreparedBackendManifest &manifest) noexcept {
  std::uint64_t dependency_stages = 0u;
  bool dependencies_complete = manifest.source_library_dependency_count != 0u &&
                               manifest.cache_dependency_entry_count ==
                                   manifest.source_library_dependency_count;
  for (std::size_t index = 0u; index < manifest.cache_dependency_entry_count;
       ++index) {
    const PreparedBackendCacheDependency &dependency =
        manifest.source_dependencies[index];
    dependencies_complete =
        dependencies_complete && dependency.complete() &&
        rund::kernel::checked::add(dependency_stages,
                                   dependency.pipeline_stage_count,
                                   dependency_stages);
  }
  manifest.source_dependencies_complete = dependencies_complete;
  if (manifest.source_build_count == 0u ||
      manifest.pipeline_stage_count == 0u ||
      manifest.source_library_dependency_count == 0u ||
      manifest.capture_binding_slot_upper == 0u ||
      !ValidPreparedBackendControlManifest(manifest) ||
      !manifest.source_dependencies_complete ||
      dependency_stages != manifest.pipeline_stage_count ||
      !rund::kernel::checked::add(manifest.pipeline_stage_count,
                                  manifest.source_library_dependency_count,
                                  manifest.cold_cache_native_object_count)) {
    manifest.ok = false;
    manifest.reason = "compute_pipeline_capacity";
    return false;
  }
  manifest.ok = true;
  manifest.reason = "ok";
  return true;
}

} // namespace rund::node::accel::detail
