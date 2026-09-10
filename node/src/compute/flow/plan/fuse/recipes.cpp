#include "internal.hpp"

#include <array>
#include <limits>
#include <span>

namespace rund::compute::detail::flow_fuse {

[[nodiscard]] Status
build_recipes(const FlowState &flow, const std::vector<bool> &keep,
              const std::vector<StepLivePlan> &map_plans,
              const std::vector<ExpressionGroupPlan> &expression_plans,
              std::vector<MapRecipe> &recipes) {
  recipes.assign(flow.steps.size(), MapRecipe{});
  for (std::size_t step_index = 0u; step_index < flow.steps.size();
       ++step_index) {
    if (!keep[step_index]) {
      continue;
    }
    const auto *const map = std::get_if<MapStep>(&flow.steps[step_index]);
    if (map == nullptr) {
      continue;
    }
    const std::span<const std::uint32_t> map_inputs =
        flow.value_ids.view(map->inputs);
    const std::span<const std::uint32_t> map_outputs =
        flow.value_ids.view(map->outputs);
    const StepLivePlan &plan = map_plans[step_index];
    MapRecipe &recipe = recipes[step_index];
    recipe.name = map->name;
    recipe.control = map->control;
    std::array<std::uint32_t, MaxMapInputs> input_map{};
    input_map.fill(std::numeric_limits<std::uint32_t>::max());
    recipe.inputs.reserve(map_inputs.size());
    recipe.outputs.reserve(map_outputs.size());
    for (std::size_t input = 0u; input < map_inputs.size(); ++input) {
      if ((plan.used_inputs & live_bit(input)) == 0u) {
        continue;
      }
      input_map[input] = static_cast<std::uint32_t>(recipe.inputs.size());
      recipe.inputs.push_back(map_inputs[input]);
    }
    for (std::size_t output = 0u; output < map_outputs.size(); ++output) {
      if ((plan.live_outputs & live_bit(output)) != 0u) {
        recipe.outputs.push_back(map_outputs[output]);
      }
    }
    if (recipe.outputs.empty() ||
        !project_expressions(map->expressions, plan, expression_plans,
                             input_map, recipe.expressions) ||
        recipe.outputs.size() != recipe.expressions.size()) {
      return Status::fail(Reason::ExpressionCapacity);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::flow_fuse
