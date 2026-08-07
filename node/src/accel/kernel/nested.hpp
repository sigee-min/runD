#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

enum class NestedTemplatePhase : std::uint8_t {
  Seed,
  Action,
  Fold,
};

struct NestedTemplateRouteProjection final {
  std::size_t template_index{};
  std::uint64_t occurrence_count{};
  std::uint32_t iteration{};
  std::uint32_t bound{};
  std::uint32_t outer_iteration{};
  std::uint32_t outer_bound{};
  std::uint32_t inner_iteration{};
  std::uint32_t inner_bound{};
  std::uint32_t inner_advance{};
  std::uint32_t route{};
  NestedTemplatePhase phase{NestedTemplatePhase::Seed};

  [[nodiscard]] constexpr bool
  operator==(const NestedTemplateRouteProjection &) const noexcept = default;
};

// Pointer-free, constructor-closed authority for one compact nested route
// group. Every phase span, retained parity owner, authored occurrence count,
// and outer-to-Fold route choice is projected from this value.
class NestedTemplateShape final {
public:
  constexpr NestedTemplateShape() noexcept = default;

  [[nodiscard]] constexpr std::size_t first() const noexcept { return first_; }
  [[nodiscard]] constexpr std::size_t seed_first() const noexcept {
    return first_;
  }
  [[nodiscard]] constexpr std::size_t action_first() const noexcept {
    return action_first_;
  }
  [[nodiscard]] constexpr std::size_t fold_first() const noexcept {
    return fold_first_;
  }
  [[nodiscard]] constexpr std::size_t end() const noexcept { return end_; }
  [[nodiscard]] constexpr std::uint32_t outer_bound() const noexcept {
    return outer_bound_;
  }
  [[nodiscard]] constexpr std::uint32_t seed_count() const noexcept {
    return outer_bound_;
  }
  [[nodiscard]] constexpr std::uint32_t inner_bound() const noexcept {
    return inner_bound_;
  }
  [[nodiscard]] constexpr std::uint32_t action_count() const noexcept {
    return inner_bound_;
  }
  [[nodiscard]] constexpr std::uint32_t fold_count() const noexcept {
    return valid() ? static_cast<std::uint32_t>(end_ - fold_first_) : 0u;
  }
  [[nodiscard]] constexpr std::uint64_t compact_entry_count() const noexcept {
    return compact_entry_count_;
  }
  [[nodiscard]] constexpr std::uint64_t retained_entry_count() const noexcept {
    return retained_entry_count_;
  }
  [[nodiscard]] constexpr std::uint64_t
  authored_occurrence_count() const noexcept {
    return authored_occurrence_count_;
  }
  [[nodiscard]] constexpr std::uint64_t
  authored_seed_occurrence_count() const noexcept {
    return outer_bound_;
  }
  [[nodiscard]] constexpr std::uint64_t
  authored_action_occurrence_count() const noexcept {
    return static_cast<std::uint64_t>(outer_bound_) * inner_bound_;
  }
  [[nodiscard]] constexpr std::uint64_t
  authored_fold_occurrence_count() const noexcept {
    return outer_bound_;
  }
  [[nodiscard]] constexpr std::uint64_t
  transduced_occurrence_count() const noexcept {
    return static_cast<std::uint64_t>(outer_bound_) * 3u;
  }
  [[nodiscard]] constexpr bool action_group_candidate() const noexcept {
    return inner_bound_ > 1u;
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return outer_bound_ != 0u && end_ > first_ &&
           compact_entry_count_ == end_ - first_;
  }

  [[nodiscard]] constexpr bool
  project(const std::size_t template_index,
          NestedTemplateRouteProjection &out) const noexcept {
    out = {};
    if (!valid() || template_index < first_ || template_index >= end_) {
      return false;
    }
    out.template_index = template_index;
    out.outer_bound = outer_bound_;
    out.inner_bound = inner_bound_;
    if (template_index < action_first_) {
      const auto iteration =
          static_cast<std::uint32_t>(template_index - first_);
      out.occurrence_count = 1u;
      out.iteration = iteration;
      out.bound = outer_bound_;
      out.outer_iteration = iteration;
      out.phase = NestedTemplatePhase::Seed;
      return true;
    }
    if (template_index < fold_first_) {
      const auto iteration =
          static_cast<std::uint32_t>(template_index - action_first_);
      out.occurrence_count = outer_bound_;
      out.iteration = iteration;
      out.bound = inner_bound_;
      out.inner_iteration = iteration;
      out.inner_advance = 1u;
      out.phase = NestedTemplatePhase::Action;
      return true;
    }
    const auto route = static_cast<std::uint32_t>(template_index - fold_first_);
    out.iteration = route;
    out.bound = 3u;
    out.route = route;
    out.phase = NestedTemplatePhase::Fold;
    out.occurrence_count = route == 0u   ? 1u
                           : route == 1u ? outer_bound_ / 2u
                                         : (outer_bound_ - 1u) / 2u;
    return true;
  }

  [[nodiscard]] constexpr bool
  fold_route_for_outer(const std::uint32_t outer,
                       std::uint32_t &route) const noexcept {
    route = 0u;
    if (!valid() || outer >= outer_bound_) {
      return false;
    }
    route = outer == 0u ? 0u : (outer & 1u) != 0u ? 1u : 2u;
    return true;
  }

  // Action parity is the only compact route family eligible to borrow a
  // prior immutable Job/template owner. Binding/program equality remains a
  // separate consumer proof.
  [[nodiscard]] constexpr bool
  retained_owner(const std::size_t template_index,
                 std::size_t &owner) const noexcept {
    owner = template_index;
    NestedTemplateRouteProjection route{};
    if (!project(template_index, route)) {
      return false;
    }
    if (route.phase == NestedTemplatePhase::Action && route.iteration >= 2u) {
      owner = action_first_ + (route.iteration & 1u);
    }
    return true;
  }

  [[nodiscard]] constexpr bool
  operator==(const NestedTemplateShape &) const noexcept = default;

private:
  constexpr NestedTemplateShape(
      const std::size_t first, const std::size_t action_first,
      const std::size_t fold_first, const std::size_t end,
      const std::uint32_t outer_bound, const std::uint32_t inner_bound,
      const std::uint64_t compact_entry_count,
      const std::uint64_t retained_entry_count,
      const std::uint64_t authored_occurrence_count) noexcept
      : first_{first}, action_first_{action_first}, fold_first_{fold_first},
        end_{end}, outer_bound_{outer_bound}, inner_bound_{inner_bound},
        compact_entry_count_{compact_entry_count},
        retained_entry_count_{retained_entry_count},
        authored_occurrence_count_{authored_occurrence_count} {}

  std::size_t first_{};
  std::size_t action_first_{};
  std::size_t fold_first_{};
  std::size_t end_{};
  std::uint32_t outer_bound_{};
  std::uint32_t inner_bound_{};
  std::uint64_t compact_entry_count_{};
  std::uint64_t retained_entry_count_{};
  std::uint64_t authored_occurrence_count_{};

  friend constexpr bool
  ProveNestedTemplateShape(std::size_t first, std::uint32_t maximum,
                           std::uint32_t tile, std::uint32_t inner_bound,
                           NestedTemplateShape &out) noexcept;
};

[[nodiscard]] constexpr bool
ProveNestedTemplateShape(const std::size_t first, const std::uint32_t maximum,
                         const std::uint32_t tile,
                         const std::uint32_t inner_bound,
                         NestedTemplateShape &out) noexcept {
  out = {};
  if (maximum == 0u || tile == 0u || tile > maximum) {
    return false;
  }
  const std::uint64_t outer =
      (static_cast<std::uint64_t>(maximum) + tile - 1u) / tile;
  const std::uint64_t compact = outer + inner_bound + 3u;
  const std::uint64_t retained =
      outer + std::min<std::uint32_t>(inner_bound, 2u) + 3u;
  const std::uint64_t occurrences =
      outer * (static_cast<std::uint64_t>(inner_bound) + 2u);
  if (outer == 0u || outer > std::numeric_limits<std::uint32_t>::max() ||
      compact > std::numeric_limits<std::size_t>::max() - first) {
    return false;
  }
  const std::size_t action_first = first + static_cast<std::size_t>(outer);
  if (inner_bound > std::numeric_limits<std::size_t>::max() - action_first) {
    return false;
  }
  const std::size_t fold_first = action_first + inner_bound;
  if (3u > std::numeric_limits<std::size_t>::max() - fold_first) {
    return false;
  }
  out = NestedTemplateShape{
      first,
      action_first,
      fold_first,
      fold_first + 3u,
      static_cast<std::uint32_t>(outer),
      inner_bound,
      compact,
      retained,
      occurrences,
  };
  return true;
}

} // namespace rund::node::accel::detail
