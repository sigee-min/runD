#include "local.hpp"

#include <cstddef>
#include <string_view>

namespace rund::node::accel::detail::recurrence_source_detail {
namespace {

[[nodiscard]] bool AddEvent(RecurrenceSourceRecipe &recipe,
                            const SourceEvent event) noexcept {
  if (recipe.event_count == recipe.events.size()) {
    return false;
  }
  recipe.events[recipe.event_count++] = event;
  return true;
}

[[nodiscard]] bool
InputNestedInOutput(const SourceBinding &input,
                    const std::span<const OutputBinding> outputs) noexcept {
  bool nested = false;
  for (const OutputBinding &output : outputs) {
    const bool overlaps = input.load_begin < output.store_end &&
                          input.load_end > output.store_begin;
    if (!overlaps) {
      continue;
    }
    if (nested || input.load_begin < output.value_begin ||
        input.load_end > output.value_end) {
      return false;
    }
    nested = true;
  }
  return nested;
}

} // namespace

bool PopulateRecipeEvents(RecurrenceSourceRecipe &recipe) noexcept {
  std::size_t variant = 0u;
  constexpr std::string_view Variant = "// artifact_variant=canonical";
  if (!FindOne(recipe.source, Variant, variant) ||
      !AddEvent(recipe, SourceEvent{.begin = variant,
                                    .end = variant + Variant.size(),
                                    .kind = SourceEventKind::Variant})) {
    return false;
  }
  if (recipe.api == ComputeApi::Metal) {
    std::size_t name_begin = 0u;
    std::size_t name_end = 0u;
    if (!FindMetalKernelName(recipe.source, recipe, name_begin, name_end) ||
        !AddEvent(recipe, SourceEvent{.begin = name_begin,
                                      .end = name_end,
                                      .kind = SourceEventKind::MetalName})) {
      return false;
    }
  }

  std::size_t body_begin = 0u;
  std::size_t body_end = 0u;
  if (recipe.api == ComputeApi::Metal) {
    constexpr std::string_view Gid =
        "    uint gid [[thread_position_in_grid]]) {\n";
    if (!FindOne(recipe.source, Gid, body_begin)) {
      return false;
    }
    body_end = body_begin + Gid.size();
  } else {
    constexpr std::string_view Dispatch =
        "layout(push_constant) uniform RundDispatch {\n"
        "  uint tile_count;\n"
        "  uint iterations;\n"
        "} rund_dispatch;\n";
    constexpr std::string_view Anchor =
        "  if (gid >= rund_dispatch.tile_count) { return; }\n";
    std::size_t declaration = 0u;
    if (!FindOne(recipe.source, Dispatch, declaration) ||
        !FindOne(recipe.source, Anchor, body_begin)) {
      return false;
    }
    body_begin += Anchor.size();
    body_end = body_begin;
  }
  if (!AddEvent(recipe, SourceEvent{.begin = body_begin,
                                    .end = body_end,
                                    .kind = SourceEventKind::Body})) {
    return false;
  }
  const std::span<const OutputBinding> outputs{recipe.outputs.data(),
                                               recipe.output_count};
  for (std::size_t index = 0u; index < recipe.input_count; ++index) {
    const SourceBinding &input = recipe.inputs[index];
    const bool nested = InputNestedInOutput(input, outputs);
    if ((!nested &&
         !AddEvent(recipe,
                   SourceEvent{.begin = input.load_begin,
                               .end = input.load_end,
                               .index = static_cast<std::uint8_t>(index),
                               .kind = SourceEventKind::Input})) ||
        (nested && input.load_begin == input.load_end)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < recipe.output_count; ++index) {
    const OutputBinding &output = recipe.outputs[index];
    if (!AddEvent(recipe, SourceEvent{.begin = output.store_begin,
                                      .end = output.store_end,
                                      .index = static_cast<std::uint8_t>(index),
                                      .kind = SourceEventKind::Output})) {
      return false;
    }
  }
  const std::size_t epilogue = recipe.source.rfind("}\n");
  return epilogue != std::string_view::npos && epilogue >= body_end &&
         AddEvent(recipe, SourceEvent{.begin = epilogue,
                                      .end = epilogue,
                                      .kind = SourceEventKind::Epilogue});
}

} // namespace rund::node::accel::detail::recurrence_source_detail
