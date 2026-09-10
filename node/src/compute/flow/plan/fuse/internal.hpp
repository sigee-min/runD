#pragma once

#include "../state.hpp"

#include <vector>

namespace rund::compute::detail::flow_fuse {

enum class Compose : unsigned char {
  Applied,
  Limit,
  Invalid,
};

[[nodiscard]] Compose compose(const MapRecipe &producer, MapRecipe &consumer);

[[nodiscard]] Status
build_recipes(const FlowState &flow, const std::vector<bool> &keep,
              const std::vector<StepLivePlan> &map_plans,
              const std::vector<ExpressionGroupPlan> &expression_plans,
              std::vector<MapRecipe> &recipes);

} // namespace rund::compute::detail::flow_fuse
