#include "../source.hpp"

#include "../../backend/exception.hpp"
#include "../../backend/source/storage.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace rund::node::accel::detail {

bool MaterializeMapRecurrenceArtifact(
    const rund::kernel::LoweringArtifact &canonical,
    const MapRecurrenceSourcePlan &plan, const std::uint64_t input_count,
    const std::uint64_t output_count,
    const std::span<const std::uint64_t> history_pitch_bytes,
    rund::kernel::LoweringArtifact &artifact,
    const std::uint64_t source_reserve_upper_bytes) {
  const recurrence_source_detail::RecurrenceSourceRecipe recipe =
      recurrence_source_detail::BuildRecipe(canonical, input_count,
                                            output_count,
                                            history_pitch_bytes);
  const MapRecurrenceSourcePlan observed = recurrence_source_detail::PlanRecipe(
      canonical, recipe);
  if (!plan.ok || !observed.ok || !recipe.ok ||
      plan.exact_source_bytes != observed.exact_source_bytes ||
      plan.source_upper_bytes != observed.source_upper_bytes ||
      plan.source_storage_upper_bytes != observed.source_storage_upper_bytes ||
      plan.metadata_storage_upper_bytes !=
          observed.metadata_storage_upper_bytes ||
      plan.history != observed.history) {
    return false;
  }
  const std::uint64_t reserve_upper =
      std::max(plan.source_upper_bytes, source_reserve_upper_bytes);
  std::uint64_t reserve_storage_upper = 0u;
  std::uint64_t materialization_host_upper = 0u;
  if (!backend_source_recipe::string_external_storage_upper_bytes(
          reserve_upper, reserve_storage_upper) ||
      !rund::kernel::checked::add(plan.metadata_storage_upper_bytes,
                                  reserve_storage_upper,
                                  materialization_host_upper)) {
    return false;
  }
  try {
    rund::kernel::LoweringArtifact transformed{
        .key = recurrence_source_detail::RecurrenceKey(canonical.key,
                                                        plan.history),
        .kind = canonical.kind,
        .metadata = canonical.metadata,
        .source_text_upper_bytes = plan.source_upper_bytes,
        .ok = canonical.ok,
        .reason = canonical.reason,
    };
    transformed.source_text = backend_source_recipe::materialize(
        recipe, plan.exact_source_bytes, reserve_upper);
    if (transformed.source_text.empty() ||
        transformed.source_text.size() != plan.exact_source_bytes ||
        !backend_source_recipe::string_external_storage_within(
            transformed.source_text, reserve_storage_upper) ||
        transformed.retained_dynamic_memory_bytes() >
            materialization_host_upper) {
      return false;
    }
    transformed.metadata.map.op_hash_hi = transformed.key.op_hash_hi;
    transformed.metadata.map.op_hash_lo = transformed.key.op_hash_lo;
    artifact = std::move(transformed);
    return true;
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return false;
  }
}

[[nodiscard]] bool
TransformSource(rund::kernel::LoweringArtifact &artifact,
                const std::uint64_t input_count,
                const std::uint64_t output_count,
                const std::span<const std::uint64_t> history_pitch_bytes) {
  const MapRecurrenceSourcePlan plan = PlanMapRecurrenceSource(
      artifact, input_count, output_count, history_pitch_bytes);
  rund::kernel::LoweringArtifact transformed{};
  if (!plan.ok || !MaterializeMapRecurrenceArtifact(
                      artifact, plan, input_count, output_count,
                      history_pitch_bytes, transformed)) {
    return false;
  }
  artifact = std::move(transformed);
  return true;
}

} // namespace rund::node::accel::detail
