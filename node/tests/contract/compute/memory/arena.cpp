#include "local.hpp"

#include "../../../../src/compute/graph/state.hpp"

#include "../allocation.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace rund_node_memory_contract {

int CheckValueRouteArena() {
  using rund::compute::detail::ValueIdArena;
  using rund::compute::detail::ValueRoutes;
  constexpr std::size_t RouteCount = 1024u;
  std::array<ValueRoutes, RouteCount> routes{};
  ValueIdArena arena;
  constexpr std::array<std::uint32_t, 1u> one{1u};
  ValueIdArena shape_arena;
  const std::optional<ValueRoutes> zero_input = shape_arena.store({}, one);
  if (!zero_input) {
    return 1;
  }
  const std::span<const std::uint32_t> stored_output =
      shape_arena.view(zero_input->outputs);
  if (!shape_arena.valid(zero_input->inputs) ||
      zero_input->inputs.count != 0u || stored_output.size() != one.size() ||
      stored_output.front() != one.front() || shape_arena.store(one, {}) ||
      shape_arena.size() != 1u) {
    return 1;
  }
  node_compute_allocation::FailNext();
  try {
    (void)arena.store(one, one);
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
    const std::optional<ValueRoutes> stored = arena.store(inputs, outputs);
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
      arena.store(arena.view(routes.front().inputs), one) ||
      arena.size() != before_alias) {
    return 7;
  }
  return 0;
}

} // namespace rund_node_memory_contract
