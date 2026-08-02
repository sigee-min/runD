#pragma once

#include "../../../kernel/backend/source_recipe.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanFixedSourceArtifact(const std::string_view source) {
  const auto recipe = [source]<typename Sink>(Sink &sink) noexcept(
                          noexcept(sink.append(std::string_view{}))) {
    return sink.append(source);
  };
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text =
      backend_source_recipe::materialize(recipe, source.size());
  artifact.source_text_upper_bytes = source.size();
  artifact.ok = !artifact.source_text.empty();
  artifact.reason = artifact.ok ? "ok" : "compute_pipeline_capacity";
  return artifact;
}

} // namespace rund::node::accel::detail
