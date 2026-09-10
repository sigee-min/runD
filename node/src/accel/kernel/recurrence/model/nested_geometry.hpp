#pragma once

#include "map.hpp"

#include <type_traits>

namespace rund::node::accel::detail {

enum class NestedAggregateState : std::uint8_t { Ineligible, Ready, Invalid };
enum class NestedAggregateKind : std::uint8_t { WindowIndexedReduceSumU32 };
enum class NestedScalarValue : std::uint8_t {
  None,
  TileState,
  TileCount,
  OuterState,
  Immediate,
};
enum class NestedScalarOp : std::uint8_t { None, AddWrapU32 };

struct NestedScalarExpr final {
  NestedScalarOp op{NestedScalarOp::None};
  NestedScalarValue lhs{NestedScalarValue::None};
  NestedScalarValue rhs{NestedScalarValue::None};
  std::uint32_t immediate{};
};

class NestedTemplateGeometry;

template <class Entry>
[[nodiscard]] bool
ProveNestedTemplateGeometry(std::span<const Entry> templates, std::size_t first,
                            NestedTemplateGeometry &out) noexcept;

class NestedTemplateGeometry final {
public:
  constexpr NestedTemplateGeometry() noexcept = default;
  [[nodiscard]] constexpr const NestedTemplateShape &shape() const noexcept {
    return shape_;
  }
  [[nodiscard]] constexpr std::size_t first() const noexcept {
    return shape_.first();
  }
  [[nodiscard]] constexpr std::size_t action_first() const noexcept {
    return shape_.action_first();
  }
  [[nodiscard]] constexpr std::size_t fold_first() const noexcept {
    return shape_.fold_first();
  }
  [[nodiscard]] constexpr std::size_t end() const noexcept {
    return shape_.end();
  }
  [[nodiscard]] constexpr std::uint32_t outer_bound() const noexcept {
    return shape_.outer_bound();
  }
  [[nodiscard]] constexpr std::uint32_t inner_bound() const noexcept {
    return shape_.inner_bound();
  }
  [[nodiscard]] constexpr std::uint32_t logical_step() const noexcept {
    return logical_step_;
  }
  [[nodiscard]] constexpr const BackendWindow *window() const noexcept {
    return window_;
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return window_ != nullptr && shape_.valid();
  }
  [[nodiscard]] constexpr bool proves_action_span(
      const std::span<const BackendBatchEntry> entries) const noexcept {
    return action_entries_ != nullptr && entries.data() == action_entries_ &&
           entries.size() == shape_.inner_bound();
  }

private:
  constexpr NestedTemplateGeometry(
      const NestedTemplateShape shape, const std::uint32_t logical_step,
      const BackendWindow *const window,
      const BackendBatchEntry *const action_entries) noexcept
      : shape_{shape}, logical_step_{logical_step}, window_{window},
        action_entries_{action_entries} {}

  NestedTemplateShape shape_{};
  std::uint32_t logical_step_{};
  const BackendWindow *window_{};
  const BackendBatchEntry *action_entries_{};

  template <class Entry>
  friend bool ProveNestedTemplateGeometry(std::span<const Entry> templates,
                                          std::size_t first,
                                          NestedTemplateGeometry &out) noexcept;
};

namespace nested_template_detail {

[[nodiscard]] inline const BackendRecurrence &
recurrence(const BackendRecurrence &value) noexcept {
  return value;
}
[[nodiscard]] inline const BackendRecurrence &
recurrence(const BackendBatchEntry &value) noexcept {
  return value.recurrence;
}

[[nodiscard]] inline bool same_read(const BackendRead &left,
                                    const BackendRead &right) noexcept {
  const auto &a = left.source;
  const auto &b = right.source;
  return a.id == b.id && a.bytes == b.bytes &&
         a.offset_bytes == b.offset_bytes &&
         a.element_bytes == b.element_bytes &&
         a.stride_bytes == b.stride_bytes && a.count == b.count &&
         a.usage == b.usage && left.handle == right.handle;
}

[[nodiscard]] inline bool same_window(const BackendWindow &left,
                                      const BackendWindow &right) noexcept {
  if (left.maximum != right.maximum || left.tile != right.tile ||
      left.expected != right.expected || left.state != right.state ||
      left.outer_bound != right.outer_bound ||
      left.inner_bound != right.inner_bound ||
      left.has_terminal != right.has_terminal ||
      !same_read(left.count, right.count)) {
    return false;
  }
  for (std::size_t bank = 0u; bank < left.terminal.size(); ++bank) {
    if (left.has_terminal &&
        !same_read(left.terminal[bank], right.terminal[bank])) {
      return false;
    }
  }
  return true;
}

} // namespace nested_template_detail

[[nodiscard]] inline constexpr bool ProjectNestedBackendWindowPhase(
    const NestedTemplatePhase phase, BackendWindowPhase &backend) noexcept {
  switch (phase) {
  case NestedTemplatePhase::Seed:
    backend = BackendWindowPhase::NestedSeed;
    return true;
  case NestedTemplatePhase::Action:
    backend = BackendWindowPhase::NestedAction;
    return true;
  case NestedTemplatePhase::Fold:
    backend = BackendWindowPhase::NestedFold;
    return true;
  }
  return false;
}

struct NestedTemplateRecurrenceIdentityBase final {
  std::uint32_t logical_step{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t expected{};
  std::uint32_t state{};
  bool has_terminal{};
};

[[nodiscard]] inline constexpr bool ProjectNestedRecurrenceIdentity(
    const NestedTemplateShape &shape, const std::size_t template_index,
    const NestedTemplateRecurrenceIdentityBase base,
    PreparedKernelRecurrenceIdentity &out) noexcept {
  out = {};
  NestedTemplateRouteProjection route{};
  if (!shape.project(template_index, route)) {
    return false;
  }
  BackendWindowPhase phase{};
  if (!ProjectNestedBackendWindowPhase(route.phase, phase)) {
    return false;
  }
  out = PreparedKernelRecurrenceIdentity{
      .logical_step = base.logical_step,
      .iteration = route.iteration,
      .bound = route.bound,
      .maximum = base.maximum,
      .tile = base.tile,
      .expected = base.expected,
      .outer_iteration = route.outer_iteration,
      .outer_bound = route.outer_bound,
      .inner_iteration = route.inner_iteration,
      .inner_bound = route.inner_bound,
      .route = route.route,
      .state = base.state,
      .phase = phase,
      .writes_each_iteration = false,
      .has_window = true,
      .has_terminal = base.has_terminal,
  };
  return true;
}

template <class Entry>
[[nodiscard]] bool
ProveNestedTemplateGeometry(const std::span<const Entry> templates,
                            const std::size_t first,
                            NestedTemplateGeometry &out) noexcept {
  out = {};
  if (first >= templates.size()) {
    return false;
  }
  const BackendRecurrence &source_recurrence =
      nested_template_detail::recurrence(templates[first]);
  const BackendWindow *const source = source_recurrence.window;
  if (source == nullptr || source->phase != BackendWindowPhase::NestedSeed ||
      source->maximum == 0u || source->tile == 0u ||
      source->tile > source->maximum || source->outer_bound == 0u) {
    return false;
  }
  NestedTemplateShape shape{};
  if (!ProveNestedTemplateShape(first, source->maximum, source->tile,
                                source->inner_bound, shape) ||
      shape.outer_bound() != source->outer_bound ||
      shape.end() > templates.size()) {
    return false;
  }
  for (std::size_t index = shape.first(); index < shape.end(); ++index) {
    const BackendRecurrence &current_recurrence =
        nested_template_detail::recurrence(templates[index]);
    const BackendWindow *const current = current_recurrence.window;
    NestedTemplateRouteProjection route{};
    BackendWindowPhase phase{};
    if (current == nullptr || current_recurrence.writes_each_iteration ||
        !nested_template_detail::same_window(*source, *current) ||
        current_recurrence.logical_step != source_recurrence.logical_step ||
        !shape.project(index, route) ||
        !ProjectNestedBackendWindowPhase(route.phase, phase) ||
        current->phase != phase ||
        current_recurrence.iteration != route.iteration ||
        current_recurrence.bound != route.bound ||
        current->outer_iteration != route.outer_iteration ||
        current->outer_bound != route.outer_bound ||
        current->inner_iteration != route.inner_iteration ||
        current->inner_bound != route.inner_bound ||
        current->inner_advance != route.inner_advance ||
        current->route != route.route) {
      return false;
    }
  }
  const BackendBatchEntry *action_entries = nullptr;
  if constexpr (std::is_same_v<Entry, BackendBatchEntry>) {
    action_entries = templates.data() + shape.action_first();
  }
  out = NestedTemplateGeometry{shape, source_recurrence.logical_step, source,
                               action_entries};
  return true;
}

} // namespace rund::node::accel::detail
