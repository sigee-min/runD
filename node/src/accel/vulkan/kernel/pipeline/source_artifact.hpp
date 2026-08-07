#pragma once

#include "../../../kernel/backend/source_recipe.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <cstdint>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {

template <typename Emit>
[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanSourceArtifact(Emit &&emit) {
  std::uint64_t source_bytes = 0u;
  const auto count = [&](backend_source_recipe::CountSink &sink) noexcept {
    return emit(sink);
  };
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  if (!backend_source_recipe::bytes(count, source_bytes)) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.source_text = backend_source_recipe::materialize(
      std::forward<Emit>(emit), source_bytes);
  artifact.source_text_upper_bytes = source_bytes;
  artifact.ok = !artifact.source_text.empty();
  artifact.reason = artifact.ok ? "ok" : "compute_pipeline_capacity";
  return artifact;
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanFixedSourceArtifact(const std::string_view source) {
  const auto recipe = [source]<typename Sink>(Sink &sink) noexcept(
                          noexcept(sink.append(std::string_view{}))) {
    return sink.append(source);
  };
  return VulkanSourceArtifact(recipe);
}

} // namespace rund::node::accel::detail
