#include "local.hpp"

#include "../../../backend/exception.hpp"
#include "../../../backend/source/edit.hpp"
#include "../../../backend/source/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] rund::kernel::LoweringArtifact
SpecializeMap(const rund::kernel::LoweringArtifact &source,
              const rund::kernel::ComputePlan &plan,
              const rund::kernel::BindingSet &bindings,
              const std::uint64_t alignment,
              const std::uint64_t reserve_upper) {
  // Specialization only mutates source text. Metadata and canonical IR remain
  // owned by the admitted Program; copying those vectors here would recreate
  // the cold intermediate layer removed by this boundary.
  const MapSourceSpecialization specialization = PlanMapSourceSpecialization(
      source, plan, bindings, alignment, reserve_upper);
  rund::kernel::LoweringArtifact artifact{
      .key = source.key,
      .kind = source.kind,
      .source_text_upper_bytes = specialization.source_upper_bytes,
      .ok = specialization.ok,
      .reason = specialization.reason,
  };
  if (!specialization.ok) {
    return artifact;
  }
  using Recipe = backend_source_recipe::BasicSourceEditRecipe<MapSourceEdit>;
  const Recipe recipe{source.source_text, specialization.active_edits()};
  artifact.source_text = backend_source_recipe::materialize(
      recipe, specialization.exact_source_bytes,
      specialization.reserve_upper_bytes);
  if (artifact.source_text.empty() ||
      artifact.source_text.size() != specialization.exact_source_bytes) {
    artifact.ok = false;
    artifact.reason = "compute_pipeline_capacity";
  }
  return artifact;
}

// Recurrence materialization owns its artifact outright and reserves the
// final backend source envelope before this call. Apply binding edits from
// right to left inside that sole allocation, then discard semantic metadata
// before compilation. Insufficient frozen capacity fails closed instead of
// allocating a second transformed source owner.
[[nodiscard]] rund::kernel::LoweringArtifact
SpecializeMapInPlace(rund::kernel::LoweringArtifact &&source,
                     const rund::kernel::ComputePlan &plan,
                     const rund::kernel::BindingSet &bindings,
                     const std::uint64_t alignment,
                     const std::uint64_t reserve_upper) {
  const MapSourceSpecialization specialization = PlanMapSourceSpecialization(
      source, plan, bindings, alignment, reserve_upper);
  if (!specialization.ok) {
    source.ok = false;
    source.reason = specialization.reason;
    return std::move(source);
  }
  std::uint64_t storage_upper = 0u;
  if (source.source_text.capacity() < specialization.reserve_upper_bytes ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          specialization.reserve_upper_bytes, storage_upper) ||
      !backend_source_recipe::string_external_storage_within(source.source_text,
                                                             storage_upper)) {
    source.ok = false;
    source.reason = "compute_pipeline_capacity";
    return std::move(source);
  }
  const std::size_t frozen_capacity = source.source_text.capacity();
  const char *const frozen_storage = source.source_text.data();
  try {
    const std::span<const MapSourceEdit> edits = specialization.active_edits();
    for (std::size_t reverse = edits.size(); reverse != 0u; --reverse) {
      const MapSourceEdit &edit = edits[reverse - 1u];
      source.source_text.replace(edit.begin, edit.end - edit.begin,
                                 edit.replacement.data(),
                                 edit.replacement_size);
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    source.ok = false;
    source.reason = "compute_pipeline_capacity";
    return std::move(source);
  }
  if (source.source_text.size() != specialization.exact_source_bytes ||
      source.source_text.capacity() != frozen_capacity ||
      source.source_text.data() != frozen_storage ||
      !backend_source_recipe::string_external_storage_within(source.source_text,
                                                             storage_upper)) {
    source.ok = false;
    source.reason = "compute_pipeline_capacity";
    return std::move(source);
  }
  source.source_text_upper_bytes = specialization.source_upper_bytes;
  source.metadata = {};
  source.canonical_ir_bytes = {};
  source.ok = true;
  source.reason = "ok";
  return std::move(source);
}

} // namespace rund::node::accel::detail
