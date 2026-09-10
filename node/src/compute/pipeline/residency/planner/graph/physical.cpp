#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] constexpr bool format_less(const FixedFormat left,
                                         const FixedFormat right) noexcept {
  if (left.integer_bits != right.integer_bits) {
    return left.integer_bits < right.integer_bits;
  }
  if (left.fraction_bits != right.fraction_bits) {
    return left.fraction_bits < right.fraction_bits;
  }
  if (left.rounding != right.rounding) {
    return left.rounding < right.rounding;
  }
  if (left.overflow != right.overflow) {
    return left.overflow < right.overflow;
  }
  return left.approximation < right.approximation;
}

} // namespace

Failure assign_graph_physical_classes(GraphPlanningState &state) {
  std::array<std::size_t, TiledGraphResourceCapacity> order{};
  const auto color_order = std::span{order}.first(state.resources.size());
  for (std::size_t index = 0u; index < color_order.size(); ++index) {
    color_order[index] = index;
  }
  std::sort(color_order.begin(), color_order.end(),
            [&state](const std::size_t left, const std::size_t right) {
              const TiledGraphResource &a = state.resources[left];
              const TiledGraphResource &b = state.resources[right];
              if (a.role != b.role) {
                return a.role < b.role;
              }
              if (a.type != b.type) {
                return a.type < b.type;
              }
              if (a.format != b.format) {
                return format_less(a.format, b.format);
              }
              if (a.page_bytes != b.page_bytes) {
                return a.page_bytes < b.page_bytes;
              }
              if (a.first_stage != b.first_stage) {
                return a.first_stage < b.first_stage;
              }
              if (a.last_stage != b.last_stage) {
                return a.last_stage < b.last_stage;
              }
              return a.resource < b.resource;
            });
  std::size_t colored_count = 0u;
  for (const std::size_t index : color_order) {
    TiledGraphResource &current = state.resources[index];
    // Every conflicting predecessor forbids one color. Mark those colors
    // in one pass instead of rescanning the same prefix for each candidate.
    std::array<bool, TiledGraphResourceCapacity> forbidden{};
    for (const std::size_t prior_index : color_order.first(colored_count)) {
      const TiledGraphResource &prior = state.resources[prior_index];
      if (prior.role != current.role || prior.type != current.type ||
          prior.format != current.format ||
          prior.page_bytes != current.page_bytes) {
        continue;
      }
      const bool disjoint = prior.last_stage < current.first_stage ||
                            current.last_stage < prior.first_stage;
      // External inputs retain distinct backing owners even when disjoint.
      if (!disjoint || current.role == GraphResourceRole::Input) {
        forbidden[prior.color] = true;
      }
    }
    const auto available = std::find(forbidden.begin(), forbidden.end(), false);
    if (available == forbidden.end()) {
      return Failure::Capacity;
    }
    current.color =
        static_cast<std::uint32_t>(available - forbidden.begin());
    ++colored_count;
  }
  state.physical_classes.clear();
  state.physical_classes.reserve(state.resources.size());
  state.page_bytes = 0u;
  for (const std::size_t index : color_order) {
    TiledGraphResource &resource = state.resources[index];
    const auto found = std::find_if(
        state.physical_classes.begin(), state.physical_classes.end(),
        [&resource](const TiledGraphPhysicalClass &physical) {
          return physical.role == resource.role &&
                 physical.type == resource.type &&
                 physical.format == resource.format &&
                 physical.color == resource.color &&
                 physical.page_bytes == resource.page_bytes;
        });
    if (found != state.physical_classes.end()) {
      resource.physical_id = found->physical_id;
      continue;
    }
    if (state.physical_classes.size() >= TiledGraphResourceCapacity ||
        !kernel::checked::add(state.page_bytes, resource.page_bytes,
                              state.page_bytes)) {
      return Failure::Capacity;
    }
    const std::uint32_t physical_id =
        static_cast<std::uint32_t>(state.physical_classes.size() + 1u);
    resource.physical_id = physical_id;
    state.physical_classes.push_back(TiledGraphPhysicalClass{
        .physical_id = physical_id,
        .role = resource.role,
        .type = resource.type,
        .format = resource.format,
        .color = resource.color,
        .page_bytes = resource.page_bytes,
    });
  }
  return Failure::None;
}

} // namespace rund::compute::detail::residency
