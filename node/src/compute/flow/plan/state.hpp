#pragma once

#include "../state.hpp"

#include "../../expression/state.hpp"

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rund::compute::detail {

static_assert(MaxMapInputs <= 16u);
static_assert(MaxOutputs <= 16u);

struct MapLivePlan final {
  std::uint16_t used_inputs = 0u;
  std::uint16_t live_outputs = 0u;
  std::size_t expression_begin = 0u;
  std::uint16_t expression_groups = 0u;
};

[[nodiscard]] constexpr std::uint16_t
live_bit(const std::size_t index) noexcept {
  return static_cast<std::uint16_t>(std::uint16_t{1u} << index);
}

using ExpressionNeeded = std::bitset<ExpressionCapacity>;

struct ExpressionGroupPlan final {
  const ExprState *group = nullptr;
  ExpressionNeeded needed{};
  std::uint16_t outputs = 0u;
  std::uint16_t extent = 0u;
  std::uint16_t live_nodes = 0u;
};

struct MapRecipe final {
  std::string_view name;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> indices;
  std::vector<std::uint32_t> outputs;
  std::vector<ExprRef> expressions;
  FlowControl control{};
  bool fused{};

  [[nodiscard]] bool active() const noexcept {
    return !outputs.empty();
  }
};

[[nodiscard]] bool project_expressions(
    std::span<const ExprRef> expressions, const MapLivePlan &map_plan,
    std::span<const ExpressionGroupPlan> plans,
    std::span<const std::uint32_t> input_map, std::vector<ExprRef> &projected);

[[nodiscard]] Status
plan_liveness(const FlowState &flow, std::vector<bool> &needed,
              std::vector<bool> &keep, std::vector<MapLivePlan> &map_plans,
              std::vector<ExpressionGroupPlan> &expression_plans,
              std::size_t &live_steps);

[[nodiscard]] Status
canonical_step_order(const FlowState &flow, const std::vector<bool> &keep,
                     const std::vector<MapLivePlan> &map_plans,
                     std::size_t live_steps, std::vector<std::size_t> &order);

[[nodiscard]] Status
plan_maps(const FlowState &flow, const std::vector<bool> &keep,
          const std::vector<MapLivePlan> &map_plans,
          const std::vector<ExpressionGroupPlan> &expression_plans,
          std::span<const std::size_t> order,
          std::vector<MapRecipe> &recipes,
          std::vector<MapRecipe> &baseline,
          std::vector<std::uint8_t> &skipped);

} // namespace rund::compute::detail
