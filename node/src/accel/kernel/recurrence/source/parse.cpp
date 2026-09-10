#include "parse/local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

namespace rund::node::accel::detail::recurrence_source_detail {

using rund::kernel::LoweringArtifact;

RecurrenceSourceRecipe
BuildRecipe(const LoweringArtifact &artifact,
            const std::uint64_t expected_inputs,
            const std::uint64_t expected_outputs,
            const std::span<const std::uint64_t> history_pitch_bytes) noexcept {
  RecurrenceSourceRecipe recipe{};
  recipe.source = artifact.source_text;
  recipe.before = artifact.key;
  recipe.api = artifact.key.api;
  recipe.scalar = artifact.key.scalar;
  recipe.history = !history_pitch_bytes.empty();
  recipe.after = RecurrenceKey(recipe.before, recipe.history);
  const bool source_kind_matches =
      (recipe.api == ComputeApi::Metal &&
       artifact.kind == rund::kernel::LoweringArtifactKind::MetalSource) ||
      (recipe.api == ComputeApi::Vulkan &&
       artifact.kind == rund::kernel::LoweringArtifactKind::VulkanSource);
  if (!artifact.ok || !source_kind_matches || recipe.source.empty() ||
      artifact.source_text_upper_bytes < recipe.source.size() ||
      expected_outputs == 0u || expected_outputs > expected_inputs ||
      expected_inputs > RecurrenceBindingCapacity ||
      expected_outputs > RecurrenceBindingCapacity ||
      (recipe.history && history_pitch_bytes.size() != expected_outputs) ||
      recipe.before.variant !=
          rund::kernel::LoweringArtifactVariant::Canonical) {
    return recipe;
  }
  for (const std::uint64_t pitch : history_pitch_bytes) {
    if (pitch == 0u || pitch > std::numeric_limits<std::uint32_t>::max()) {
      return recipe;
    }
  }
  if (!SourceBindings(artifact.metadata, recipe.source, recipe.api,
                      recipe.scalar, history_pitch_bytes, recipe.inputs,
                      recipe.input_count, recipe.outputs,
                      recipe.output_count) ||
      recipe.input_count != expected_inputs ||
      recipe.output_count != expected_outputs) {
    return recipe;
  }
  const std::uint64_t scalar_bytes =
      rund::kernel::ComputeScalarBits(recipe.scalar) / 8u;
  for (std::size_t index = 0u; index < recipe.output_count; ++index) {
    if (recipe.inputs[index].uniform ||
        recipe.inputs[index].element_bytes != scalar_bytes ||
        recipe.outputs[index].element_bytes != scalar_bytes) {
      return recipe;
    }
  }
  if (!PopulateRecipeEvents(recipe)) {
    return recipe;
  }
  std::sort(recipe.events.begin(), recipe.events.begin() + recipe.event_count,
            [](const SourceEvent &left, const SourceEvent &right) noexcept {
              return left.begin < right.begin ||
                     (left.begin == right.begin && left.end < right.end);
            });
  std::size_t consumed = 0u;
  bool first = true;
  std::size_t prior = 0u;
  for (std::size_t index = 0u; index < recipe.event_count; ++index) {
    const SourceEvent &event = recipe.events[index];
    if (event.begin < consumed || event.end < event.begin ||
        event.end > recipe.source.size() || (!first && event.begin == prior)) {
      return recipe;
    }
    consumed = event.end;
    prior = event.begin;
    first = false;
  }
  recipe.ok = true;
  return recipe;
}

} // namespace rund::node::accel::detail::recurrence_source_detail
