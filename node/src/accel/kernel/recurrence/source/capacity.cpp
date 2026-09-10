#include "../source.hpp"

#include "../../backend/source/storage.hpp"
#include "internal.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace rund::node::accel::detail::recurrence_source_detail {

using rund::kernel::ExecutionMetadata;
using rund::kernel::LoweringArtifact;

namespace {

[[nodiscard]] bool AddStorageEnvelope(std::uint64_t &bytes,
                                      const std::uint64_t count,
                                      const std::uint64_t width) noexcept {
  std::uint64_t payload = 0u;
  if (!rund::kernel::checked::mul(count, width, payload)) {
    return false;
  }
  if (payload != 0u &&
      !rund::kernel::checked::add(
          payload, backend_source_recipe::StringExternalStorageSlackBytes,
          payload)) {
    return false;
  }
  return rund::kernel::checked::add(bytes, payload, bytes);
}

[[nodiscard]] bool MetadataStorageUpperBytes(const ExecutionMetadata &metadata,
                                             std::uint64_t &bytes) noexcept {
  bytes = 0u;
  if (!AddStorageEnvelope(bytes, metadata.param_storage.size(),
                          sizeof(metadata.param_storage.front())) ||
      !AddStorageEnvelope(bytes, metadata.input_element_bytes.size(),
                          sizeof(metadata.input_element_bytes.front())) ||
      !AddStorageEnvelope(bytes, metadata.output_element_bytes.size(),
                          sizeof(metadata.output_element_bytes.front())) ||
      !AddStorageEnvelope(bytes, metadata.binding_accesses.size(),
                          sizeof(metadata.binding_accesses.front())) ||
      !AddStorageEnvelope(bytes, metadata.binding_names.size(),
                          sizeof(metadata.binding_names.front())) ||
      !AddStorageEnvelope(bytes, metadata.read_routes.size(),
                          sizeof(metadata.read_routes.front()))) {
    return false;
  }
  for (const std::string &name : metadata.binding_names) {
    std::uint64_t storage = 0u;
    if (!backend_source_recipe::string_external_storage_upper_bytes(name.size(),
                                                                    storage) ||
        !rund::kernel::checked::add(bytes, storage, bytes)) {
      return false;
    }
  }
  return true;
}

} // namespace

[[nodiscard]] MapRecurrenceSourcePlan
PlanRecipe(const LoweringArtifact &artifact,
           const RecurrenceSourceRecipe &recipe) noexcept {
  MapRecurrenceSourcePlan plan{.history = recipe.history};
  if (!recipe.ok ||
      artifact.source_text_upper_bytes < artifact.source_text.size()) {
    return plan;
  }
  const auto count = [&](backend_source_recipe::CountSink &sink) noexcept {
    return recipe(sink);
  };
  const std::uint64_t inherited_upper_growth =
      artifact.source_text_upper_bytes - artifact.source_text.size();
  std::uint64_t metadata_storage = 0u;
  if (!backend_source_recipe::bytes(count, plan.exact_source_bytes) ||
      !rund::kernel::checked::add(plan.exact_source_bytes,
                                  inherited_upper_growth,
                                  plan.source_upper_bytes) ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          plan.source_upper_bytes, plan.source_storage_upper_bytes) ||
      !MetadataStorageUpperBytes(artifact.metadata, metadata_storage)) {
    plan.reason = "compute_pipeline_capacity";
    return plan;
  }
  plan.metadata_storage_upper_bytes = metadata_storage;
  plan.ok = true;
  plan.reason = "ok";
  return plan;
}

} // namespace rund::node::accel::detail::recurrence_source_detail

namespace rund::node::accel::detail {

MapRecurrenceSourcePlan PlanMapRecurrenceSource(
    const rund::kernel::LoweringArtifact &artifact,
    const std::uint64_t input_count, const std::uint64_t output_count,
    const std::span<const std::uint64_t> history_pitch_bytes) noexcept {
  const auto recipe = recurrence_source_detail::BuildRecipe(
      artifact, input_count, output_count, history_pitch_bytes);
  return recurrence_source_detail::PlanRecipe(artifact, recipe);
}

} // namespace rund::node::accel::detail
