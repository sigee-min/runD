#include "plan/graph.hpp"
#include "plan/state.hpp"
#include "recipe.hpp"

#include "../device/state.hpp"
#include "../graph/describe.hpp"
#include "../graph/state.hpp"
#include "../program/cache.hpp"
#include "../program/state.hpp"

#include <rund/compute/abi/graph.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] constexpr std::uint64_t
sat_add(const std::uint64_t left, const std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

[[nodiscard]] constexpr auto flow_score(const graph::Info &info) noexcept {
  // A compiled Flow is selected for execution, so retained physical bytes and
  // one invocation's mandatory read traffic share the same byte unit. The
  // first term is therefore a conservative one-run memory burden. Remaining
  // fields provide a total canonical order without device-specific tuning.
  return std::tuple{sat_add(info.memory.physical_bytes, info.read_bytes),
                    info.memory.physical_bytes, info.memory.allocation_count,
                    info.nodes.size(), info.memory.logical_bytes};
}

} // namespace

Result<std::shared_ptr<ProgramState>>
compile_flow(const std::shared_ptr<FlowState> &flow) {
  if (flow == nullptr) {
    return fail_flow(Reason::FlowCapacity);
  }
  if (!flow->status) {
    return fail_flow(flow->status.reason());
  }
  if (flow->steps.empty()) {
    return fail_flow(Reason::FlowEmpty);
  }

  Result<std::shared_ptr<DeviceState>> opened =
      flow->device != nullptr
          ? Result<std::shared_ptr<DeviceState>>::success(flow->device)
          : open_target(flow->target);
  if (!opened) {
    return fail_flow(opened.reason());
  }

  std::vector<bool> needed;
  std::vector<bool> keep;
  std::vector<MapLivePlan> map_plans;
  std::vector<ExpressionGroupPlan> expression_plans;
  std::size_t live_steps = 0u;
  const Status planned = plan_liveness(*flow, needed, keep, map_plans,
                                       expression_plans, live_steps);
  if (!planned) {
    return fail_flow(planned.reason());
  }
  std::vector<std::size_t> step_order;
  const Status ordered =
      canonical_step_order(*flow, keep, map_plans, live_steps, step_order);
  if (!ordered) {
    return fail_flow(ordered.reason());
  }
  std::vector<MapRecipe> map_recipes;
  std::vector<MapRecipe> baseline_recipes;
  std::vector<std::uint8_t> skipped;
  const Status mapped =
      plan_maps(*flow, keep, map_plans, expression_plans, step_order,
                map_recipes, baseline_recipes, skipped);
  if (!mapped) {
    return fail_flow(mapped.reason());
  }

  const std::shared_ptr<DeviceState> device = std::move(opened).value();
  if (flow->cache != nullptr && flow->cache->device != device) {
    return fail_flow(Reason::ProgramCacheDeviceMismatch);
  }
  auto candidate =
      materialize_graph(flow, device, step_order, map_recipes, skipped);
  const bool has_fusion =
      std::any_of(map_recipes.begin(), map_recipes.end(),
                  [](const MapRecipe &map) { return map.fused; }) ||
      std::any_of(skipped.begin(), skipped.end(),
                  [](const std::uint8_t value) { return value != 0u; });
  std::shared_ptr<GraphState> graph;
  if (!has_fusion) {
    if (!candidate) {
      return fail_flow(candidate.reason());
    }
    graph = std::move(candidate).value();
  } else {
    auto baseline =
        materialize_graph(flow, device, step_order, baseline_recipes);
    if (!baseline) {
      return fail_flow(baseline.reason());
    }
    graph = std::move(baseline).value();
    graph_detail::Description base = graph_detail::describe(graph);
    if (!base.status) {
      return fail_flow(base.status.reason());
    }
    if (candidate) {
      graph_detail::Description fused =
          graph_detail::describe(candidate.value());
      if (fused.status && flow_score(fused.info) < flow_score(base.info)) {
        graph = std::move(candidate).value();
        graph->authored_nodes = base.info.authored_nodes;
      }
    }
  }
  std::vector<Type> inputs;
  try {
    inputs.reserve(flow->inputs.size());
    for (const std::uint32_t input : flow->inputs) {
      inputs.push_back(flow->values[input - 1u].type);
    }
  } catch (const std::bad_alloc &) {
    return fail_flow(Reason::GraphCapacity);
  }
  std::vector<Type> outputs;
  try {
    outputs.reserve(graph->outputs.size());
    for (const std::uint32_t output : graph->outputs) {
      outputs.push_back(graph->values[output - 1u].type);
    }
  } catch (const std::bad_alloc &) {
    return fail_flow(Reason::GraphCapacity);
  }
  if (flow_count(flow) == 0u) {
    graph_detail::Description described = graph_detail::describe(graph);
    if (!described.status) {
      return fail_flow(described.status.reason());
    }
    ::rund::compute::graph::Info graph_info = std::move(described.info);
    const ::rund::compute::graph::Fingerprint fingerprint =
        graph_info.fingerprint;
    auto build = [graph, device, graph_info = std::move(graph_info)]() mutable
        -> Result<std::shared_ptr<ProgramState>> {
      try {
        auto program = std::make_shared<ProgramState>();
        program->device = device;
        program->name = "flow";
        for (const std::uint32_t output : graph->outputs) {
          program->output_sizes.push_back(graph->values[output - 1u].count);
          program->output_types.push_back(graph->values[output - 1u].type);
          program->output_formats.push_back(
              graph->values[output - 1u].fixed_format);
        }
        if (!graph->identity_outputs.empty()) {
          program->output_aliases.reserve(graph->identity_outputs.size());
          for (const std::uint32_t output : graph->identity_outputs) {
            const auto found =
                std::find(graph->outputs.begin(), graph->outputs.end(), output);
            if (found == graph->outputs.end()) {
              return fail_flow(Reason::GraphBindingInvalid);
            }
            program->output_aliases.push_back(static_cast<std::size_t>(
                std::distance(graph->outputs.begin(), found)));
          }
        }
        for (const std::uint32_t input : graph->inputs) {
          program->input_types.push_back(graph->values[input - 1u].type);
          program->input_sizes.push_back(graph->values[input - 1u].count);
          program->input_formats.push_back(
              graph->values[input - 1u].fixed_format);
        }
        program->bounded_input_capacities.assign(graph->inputs.size(), 0u);
        for (const BoundedInputSchema bounded : graph->bounded_inputs) {
          const auto count = std::find(graph->inputs.begin(),
                                       graph->inputs.end(), bounded.count);
          if (count == graph->inputs.end() || bounded.capacity == 0u) {
            return fail_flow(Reason::BoundedCountInvalid);
          }
          const std::size_t index = static_cast<std::size_t>(
              std::distance(graph->inputs.begin(), count));
          if (program->bounded_input_capacities[index] != 0u) {
            return fail_flow(Reason::BoundedCountInvalid);
          }
          program->bounded_input_capacities[index] = bounded.capacity;
        }
        if (!valid_input_shape(*program)) {
          return fail_flow(Reason::ProgramInputShapeInvalid);
        }
        program->graph_info = std::move(graph_info);
        program->empty_graph_hash =
            program->graph_info.fingerprint.hi ^
            std::rotl(program->graph_info.fingerprint.lo, 1);
        return Result<std::shared_ptr<ProgramState>>::success(
            std::move(program));
      } catch (const std::bad_alloc &) {
        return fail_flow(Reason::ProgramCapacity);
      }
    };
    if (flow->cache == nullptr) {
      return build();
    }
    return cached_program(flow->cache, fingerprint, std::move(build));
  }
  auto compiled = compile_graph(graph, inputs, outputs, flow->cache);
  if (!compiled) {
    return compiled;
  }
  return compiled;
}

} // namespace rund::compute::detail
