#pragma once

#include "guard.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

struct MetalPipelineSourceRecipe final {
  std::uint64_t recipe_id{};
  std::uint64_t raw_source_upper_bytes{};
  std::uint64_t entry_point_count{};
  std::uint64_t final_source_upper_bytes{};
  std::uint64_t pipeline_stage_count{};
  bool ok{};
};

[[nodiscard]] inline MetalPipelineSourceRecipe
MetalSourceRecipe(const std::uint64_t recipe_id,
                  const std::uint64_t raw_source_upper_bytes,
                  const std::uint64_t entry_point_count,
                  const std::uint64_t pipeline_stage_count) noexcept {
  MetalPipelineSourceRecipe recipe{
      .recipe_id = recipe_id,
      .raw_source_upper_bytes = raw_source_upper_bytes,
      .entry_point_count = entry_point_count,
      .pipeline_stage_count = pipeline_stage_count,
  };
  recipe.ok = recipe_id != 0u && raw_source_upper_bytes != 0u &&
              entry_point_count != 0u && pipeline_stage_count != 0u &&
              PipelinePrivateMetalSourceUpperBytes(
                  raw_source_upper_bytes, entry_point_count, true,
                  recipe.final_source_upper_bytes);
  return recipe;
}

} // namespace rund::node::accel::detail
