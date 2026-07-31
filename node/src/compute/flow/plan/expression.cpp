#include "state.hpp"

#include <array>
#include <bit>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] bool
project_expressions(const std::span<const ExprRef> expressions,
                    const MapLivePlan &map_plan,
                    const std::span<const ExpressionGroupPlan> plans,
                    const std::span<const std::uint32_t> input_map,
                    std::vector<ExprRef> &projected) {
  try {
    if (map_plan.expression_begin > plans.size() ||
        map_plan.expression_groups > plans.size() - map_plan.expression_begin) {
      return false;
    }
    std::array<ExprRef, MaxOutputs> roots{};
    // Projected expressions are bounded to 1024 nodes, so 16-bit ordinals
    // halve this cold stack table without narrowing any representable node.
    std::array<std::uint16_t, ExpressionCapacity> node_map;
    for (std::size_t group_index = map_plan.expression_begin;
         group_index < map_plan.expression_begin + map_plan.expression_groups;
         ++group_index) {
      const ExpressionGroupPlan &plan = plans[group_index];
      if (plan.group == nullptr || plan.extent == 0u || plan.live_nodes == 0u) {
        return false;
      }
      auto state = std::make_shared<ExprState>();
      state->nodes.reserve(plan.live_nodes);
      for (std::size_t index = 0u; index < plan.extent; ++index) {
        if (!plan.needed.test(index)) {
          continue;
        }
        ExprNode node = plan.group->nodes[index];
        if (node.operation == ExprOp::Input) {
          if (node.left >= input_map.size() ||
              input_map[node.left] ==
                  std::numeric_limits<std::uint32_t>::max()) {
            return false;
          }
          node.left = input_map[node.left];
        } else {
          const auto operand = [&](std::uint32_t &value) {
            if (value == 0u || value > index || !plan.needed.test(value - 1u)) {
              return false;
            }
            value = node_map[value - 1u];
            return true;
          };
          const std::uint8_t arity = expr_arity(node.operation);
          if (arity == InvalidArity || (arity >= 1u && !operand(node.left)) ||
              (arity >= 2u && !operand(node.right)) ||
              (arity == 3u && !operand(node.third))) {
            return false;
          }
        }
        state->nodes.push_back(node);
        node_map[index] = static_cast<std::uint16_t>(state->nodes.size());
      }
      std::uint16_t group_outputs = plan.outputs;
      while (group_outputs != 0u) {
        const std::size_t index =
            static_cast<std::size_t>(std::countr_zero(group_outputs));
        group_outputs &= static_cast<std::uint16_t>(group_outputs - 1u);
        if (index >= expressions.size()) {
          return false;
        }
        const ExprRef &expression = expressions[index];
        if (expression.state.get() != plan.group || expression.node == 0u ||
            expression.node > plan.extent ||
            !plan.needed.test(expression.node - 1u)) {
          return false;
        }
        const std::uint32_t root = node_map[expression.node - 1u];
        if (root == 0u) {
          return false;
        }
        roots[index] =
            ExprRef{state, root, expression.type, expression.fixed_format};
      }
    }
    projected.clear();
    projected.reserve(
        static_cast<std::size_t>(std::popcount(map_plan.live_outputs)));
    for (std::size_t index = 0u; index < expressions.size(); ++index) {
      if ((map_plan.live_outputs & live_bit(index)) != 0u) {
        projected.push_back(std::move(roots[index]));
      }
    }
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

} // namespace rund::compute::detail
