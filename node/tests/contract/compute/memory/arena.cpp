#include "local.hpp"

#include "../../../../src/compute/flow/recipe.hpp"
#include "../../../../src/compute/graph/build/model.hpp"
#include "../../../../src/compute/graph/state.hpp"

#include "../allocation.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

namespace rund_node_memory_contract {

int CheckValueRouteArena() {
  using rund::compute::detail::ValueIdArena;
  using rund::compute::detail::ValueRoutes;
  static_assert(!std::is_move_constructible_v<ValueIdArena>);
  static_assert(!std::is_move_assignable_v<ValueIdArena>);

  const auto store = [](ValueIdArena &arena,
                        const std::span<const std::uint32_t> inputs,
                        const std::span<const std::uint32_t> outputs)
      -> std::optional<ValueRoutes> {
    std::optional<ValueRoutes> routes;
    if (!arena.publish(inputs, outputs,
                       [&](const ValueRoutes value) { routes = value; })) {
      return std::nullopt;
    }
    return routes;
  };

  constexpr std::size_t RouteCount = 1024u;
  std::array<ValueRoutes, RouteCount> routes{};
  ValueIdArena arena;
  constexpr std::array<std::uint32_t, 1u> one{1u};
  ValueIdArena shape_arena;
  const std::optional<ValueRoutes> zero_input = store(shape_arena, {}, one);
  if (!zero_input) {
    return 1;
  }
  const std::span<const std::uint32_t> stored_output =
      shape_arena.view(zero_input->outputs);
  if (!shape_arena.valid(zero_input->inputs) ||
      zero_input->inputs.count != 0u || stored_output.size() != one.size() ||
      stored_output.front() != one.front() || store(shape_arena, one, {}) ||
      shape_arena.size() != 1u) {
    return 1;
  }
  node_compute_allocation::FailNext();
  try {
    (void)store(arena, one, one);
    return 2;
  } catch (const std::bad_alloc &) {
  }
  if (arena.size() != 0u) {
    return 3;
  }

  node_compute_allocation::Start();
  for (std::size_t index = 0u; index < RouteCount; ++index) {
    const std::array inputs{static_cast<std::uint32_t>(index + 1u),
                            static_cast<std::uint32_t>(index + 1001u)};
    const std::array outputs{static_cast<std::uint32_t>(index + 2001u)};
    const std::optional<ValueRoutes> stored = store(arena, inputs, outputs);
    if (!stored) {
      node_compute_allocation::Stop();
      return 4;
    }
    routes[index] = *stored;
  }
  node_compute_allocation::Stop();

  constexpr std::size_t IdCount = RouteCount * 3u;
  if (arena.size() != IdCount || arena.capacity() < arena.size() ||
      arena.capacity() > (arena.size() * 3u + 1u) / 2u ||
      node_compute_allocation::Count() != 18u) {
    return 5;
  }
  for (std::size_t index = 0u; index < RouteCount; ++index) {
    const std::span<const std::uint32_t> inputs =
        arena.view(routes[index].inputs);
    const std::span<const std::uint32_t> outputs =
        arena.view(routes[index].outputs);
    if (inputs.size() != 2u || outputs.size() != 1u ||
        inputs[0u] != index + 1u || inputs[1u] != index + 1001u ||
        outputs[0u] != index + 2001u) {
      return 6;
    }
  }
  const std::size_t before_alias = arena.size();
  if (arena.valid({std::numeric_limits<std::uint32_t>::max(), 1u}) ||
      !arena.view({std::numeric_limits<std::uint32_t>::max(), 1u}).empty() ||
      store(arena, arena.view(routes.front().inputs), one) ||
      arena.size() != before_alias) {
    return 7;
  }

  ValueIdArena transaction_arena;
  bool nested_published = true;
  try {
    (void)transaction_arena.publish(one, one, [&](const ValueRoutes) {
      nested_published = transaction_arena.publish(
          one, one, [](const ValueRoutes) noexcept {});
      throw std::bad_alloc{};
    });
    return 8;
  } catch (const std::bad_alloc &) {
  }
  if (nested_published || transaction_arena.size() != 0u ||
      !store(transaction_arena, one, one) || transaction_arena.size() != 2u) {
    return 9;
  }

  constexpr std::array<std::uint32_t, 1u> output{2u};
  rund::compute::detail::FlowState flow;
  node_compute_allocation::FailAfter(1u);
  const bool flow_failed = rund::compute::detail::append_primitive(
      flow, one, output, rund::compute::detail::Primitive::Reduce, {});
  node_compute_allocation::ClearFailure();
  if (flow_failed ||
      flow.status.reason() != rund::compute::Reason::FlowCapacity ||
      flow.value_ids.size() != 0u || !flow.steps.empty()) {
    return 10;
  }
  flow.status = rund::compute::Status::success();
  if (!rund::compute::detail::append_primitive(
          flow, one, output, rund::compute::detail::Primitive::Reduce, {}) ||
      flow.value_ids.size() != 2u || flow.steps.size() != 1u) {
    return 11;
  }

  rund::compute::detail::GraphState graph;
  node_compute_allocation::FailAfter(1u);
  const bool graph_failed =
      rund::compute::detail::graph_build_detail::append_primitive(
          graph, one, output, 2u, rund::compute::detail::Primitive::Reduce, {},
          {});
  node_compute_allocation::ClearFailure();
  if (graph_failed ||
      graph.status.reason() != rund::compute::Reason::GraphCapacity ||
      graph.value_ids.size() != 0u || !graph.steps.empty()) {
    return 12;
  }
  graph.status = rund::compute::Status::success();
  if (!rund::compute::detail::graph_build_detail::append_primitive(
          graph, one, output, 2u, rund::compute::detail::Primitive::Reduce, {},
          {}) ||
      graph.value_ids.size() != 2u || graph.steps.size() != 1u) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_memory_contract
