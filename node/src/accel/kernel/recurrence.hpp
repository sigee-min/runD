#pragma once

#include "backend/run.hpp"
#include "nested.hpp"
#include "prepared/template_registry.hpp"

#include <kernel/program/compute/limit.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace rund::node::accel::detail {

enum class MapRecurrenceState : std::uint8_t {
  Ineligible,
  Ready,
  Invalid,
};

// Proof-owned projection of caller-retained iteration history. Each output
// ref spans all authored slices from the first slice offset. The exact byte
// pitch remains explicit because source lowering and native descriptor
// materialization consume different projections of the same proved layout.
struct MapRecurrenceHistory final {
  static constexpr std::size_t Capacity =
      static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);
  std::array<rund::kernel::ResidentBufferRef, Capacity> outputs{};
  std::array<std::shared_ptr<void>, Capacity> handles{};
  std::array<std::uint64_t, Capacity> pitch_bytes{};
  std::uint32_t count{};

  [[nodiscard]] rund::kernel::ResidentBindingRange range() const noexcept {
    if (count == 0u || count > Capacity) {
      return {};
    }
    return rund::kernel::ResidentBindingRange{
        .refs = outputs.data(),
        .handles = handles.data(),
        .storage_count = count,
        .count = count,
    };
  }

  [[nodiscard]] std::span<const std::uint64_t> pitches() const noexcept {
    return count == 0u || count > Capacity
               ? std::span<const std::uint64_t>{}
               : std::span<const std::uint64_t>{pitch_bytes.data(), count};
  }
};

// Allocation-free source proof shared by public Pipeline planning and cold
// recurrence materialization. `exact_source_bytes` is the emitter result for
// the currently retained canonical source. `source_upper_bytes` carries the
// canonical artifact's already-proved decimal-literal growth envelope through
// the recurrence transform. String storage remains a separate dimension so
// callers never reinterpret text cardinality as allocator capacity.
struct MapRecurrenceSourcePlan final {
  std::uint64_t exact_source_bytes{};
  std::uint64_t source_upper_bytes{};
  std::uint64_t source_storage_upper_bytes{};
  // Ephemeral semantic metadata is copied into the one-shot artifact only
  // until backend specialization is complete. Source storage is deliberately
  // separate: the same allocation becomes the retained backend cache source
  // and must never be charged again as transient host memory.
  std::uint64_t metadata_storage_upper_bytes{};
  bool history{};
  bool ok{};
  const char *reason{"compute_pipeline_recurrence_source_invalid"};
};

// One allocation-free authority for public recurrence preflight and private
// backend preparation. Both paths normalize their Map bindings into the same
// pointer-free layout before any source string, backend template, descriptor
// arena, or route resource may be allocated. Candidate groups that are
// statically ineligible produce an admitted zero-group plan; a matched Map
// whose source recipe is invalid fails closed instead.
struct MapRecurrencePreparationPlan final {
  static constexpr std::size_t Capacity =
      static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);

  const KernelExecutionStep *authority{};
  const rund::kernel::LoweringArtifact *canonical_artifact{};
  rund::kernel::ComputePlan plan{};
  std::array<PreparedKernelProgramBindingIdentity, Capacity> inputs{};
  std::array<PreparedKernelProgramBindingIdentity, Capacity> outputs{};
  MapRecurrenceSourcePlan terminal_source{};
  MapRecurrenceSourcePlan history_source{};
  std::uint64_t window_count{};
  std::uint64_t group_count{};
  std::uint64_t history_group_count{};
  // Immutable terminal/history templates are shared across structurally
  // equal routes and transactional streams. These capacities freeze the
  // complete group demand of each shared owner; route group counts above
  // remain the one-stream execution-state demand.
  std::uint64_t terminal_template_group_capacity{};
  std::uint64_t history_template_group_capacity{};
  // Backend template binding specialization observes byte stride and the
  // offset residue at this alignment. Metal uses one, Vulkan uses the frozen
  // storage alignment. Keeping the normalization in the common plan lets
  // public planning and runtime cache matching share one equivalence law.
  std::uint64_t binding_alignment{};
  std::uint32_t input_count{};
  std::uint32_t output_count{};
  bool ok{};
  const char *reason{"compute_pipeline_recurrence_invalid"};

  [[nodiscard]] constexpr std::uint64_t terminal_group_count() const noexcept {
    return group_count >= history_group_count
               ? group_count - history_group_count
               : 0u;
  }

  [[nodiscard]] constexpr bool eligible() const noexcept {
    return ok && group_count != 0u;
  }

  [[nodiscard]] std::span<const PreparedKernelProgramBindingIdentity>
  input_layouts() const noexcept {
    return input_count > Capacity
               ? std::span<const PreparedKernelProgramBindingIdentity>{}
               : std::span<const PreparedKernelProgramBindingIdentity>{
                     inputs.data(), input_count};
  }

  [[nodiscard]] std::span<const PreparedKernelProgramBindingIdentity>
  output_layouts() const noexcept {
    return output_count > Capacity
               ? std::span<const PreparedKernelProgramBindingIdentity>{}
               : std::span<const PreparedKernelProgramBindingIdentity>{
                     outputs.data(), output_count};
  }
};

// Cold evidence for replacing a complete element-local Map recurrence with
// one backend kernel. Ineligible is not an error: the canonical prepared
// command stream remains authoritative. Invalid means the chain matched the
// recurrence semantics but its retained source could not be transformed
// exactly, so preparation must fail closed.
struct MapRecurrence final {
  MapRecurrenceState state = MapRecurrenceState::Ineligible;
  const BoundStep *first = nullptr;
  const BoundStep *last = nullptr;
  rund::kernel::BindingSet bindings{};
  // The admitted Program remains the only metadata/source authority. Common
  // proof freezes an exact allocation-free transform plan; a backend
  // materializes one minimal local artifact immediately before compilation
  // and never retains an intermediate transformed-source layer.
  const rund::kernel::LoweringArtifact *canonical_artifact{};
  MapRecurrenceSourcePlan source_plan{};
  rund::kernel::ComputePlan plan{};
  const rund::kernel::ComputeDispatchWindow *windows = nullptr;
  std::uint64_t window_count = 0u;
  std::uint64_t iterations = 0u;
  std::shared_ptr<const MapRecurrenceHistory> history{};
  const char *reason = "compute_pipeline_recurrence_ineligible";

  [[nodiscard]] constexpr bool ready() const noexcept {
    return state == MapRecurrenceState::Ready;
  }

  [[nodiscard]] constexpr bool invalid() const noexcept {
    return state == MapRecurrenceState::Invalid;
  }

  [[nodiscard]] bool writes_each_iteration() const noexcept {
    return history != nullptr;
  }
};

inline constexpr std::uint32_t NoTileTransducer =
    std::numeric_limits<std::uint32_t>::max();

// One proved nested Action subrange. The common accelerator compiler owns the
// classification; Metal and Vulkan only materialize the already-proved map
// recurrence. template_first/template_count retain the authored profile and
// failure coordinates even though one physical command represents the range.
struct TileTransducer final {
  MapRecurrence recurrence{};
  std::uint32_t template_first{};
  std::uint32_t template_count{};
};

enum class NestedAggregateState : std::uint8_t {
  Ineligible,
  Ready,
  Invalid,
};

enum class NestedAggregateKind : std::uint8_t {
  WindowIndexedReduceSumU32,
};

enum class NestedScalarValue : std::uint8_t {
  None,
  TileState,
  TileCount,
  OuterState,
  Immediate,
};

enum class NestedScalarOp : std::uint8_t {
  None,
  AddWrapU32,
};

struct NestedScalarExpr final {
  NestedScalarOp op{NestedScalarOp::None};
  NestedScalarValue lhs{NestedScalarValue::None};
  NestedScalarValue rhs{NestedScalarValue::None};
  std::uint32_t immediate{};
};

// Allocation-free canonical proof for one compact nested template group.
// Reservation, fallback expansion, and aggregate eligibility all consume this
// geometry instead of independently reconstructing K/N or phase ordinals.
// Placeholder coordinates that expansion overwrites are intentionally absent.
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

[[nodiscard]] inline constexpr BackendWindowPhase
ProjectNestedBackendWindowPhase(const NestedTemplatePhase phase) noexcept {
  switch (phase) {
  case NestedTemplatePhase::Seed:
    return BackendWindowPhase::NestedSeed;
  case NestedTemplatePhase::Action:
    return BackendWindowPhase::NestedAction;
  case NestedTemplatePhase::Fold:
    return BackendWindowPhase::NestedFold;
  }
  return BackendWindowPhase::Ordinary;
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
      .phase = static_cast<std::uint8_t>(
          ProjectNestedBackendWindowPhase(route.phase)),
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
    if (current == nullptr || current_recurrence.writes_each_iteration ||
        !nested_template_detail::same_window(*source, *current) ||
        current_recurrence.logical_step != source_recurrence.logical_step ||
        !shape.project(index, route) ||
        current->phase != ProjectNestedBackendWindowPhase(route.phase) ||
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

struct NestedAggregateRead final {
  BackendRead read{};
  std::uint64_t logical_count{};
  std::uint64_t element_bytes{};
};

// Plan-owned Program workspace that the complete aggregate may overwrite.
// The common proof retains the owner and exact dense range so a backend can
// borrow dead Seed storage without allocating a second scratch authority.
struct NestedAggregateWorkspace final {
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
};

// Exact per-Program dispatch projection for one aggregate command. Authored
// occurrence cardinality remains a NestedTemplateShape projection.
struct NestedAggregateProfileProjection final {
  std::uint32_t seed_dispatches_per_occurrence{};
  std::uint32_t action_dispatches_per_occurrence{};
  std::uint32_t fold_dispatches_per_occurrence{};
  bool aggregate_profile_supported{};
};

struct NestedAggregateFailureProjection final {
  std::uint32_t logical_step{};
  std::uint32_t invalid_index_source_node{NoNode};
  std::uint32_t reduce_overflow_source_node{NoNode};
  std::uint32_t count_overflow_reason{};
  std::uint32_t invalid_index_reason{};
  std::uint32_t reduce_overflow_reason{};
  rund::compute::PipelineNestedPhase phase{
      rund::compute::PipelineNestedPhase::None};
  bool inner_coordinate_unknown{};
};

// Backend-neutral proof object for the complete Seed/Action/Fold recurrence.
// Shape remains the only template-span/cardinality authority. Backends consume
// its projections and the normalized opcodes without rediscovering either from
// generated source or a writable span mirror.
struct NestedAggregate final {
  NestedAggregateState state{NestedAggregateState::Ineligible};
  NestedAggregateKind kind{NestedAggregateKind::WindowIndexedReduceSumU32};
  NestedTemplateShape shape{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  NestedAggregateRead queue{};
  NestedAggregateRead domain{};
  NestedAggregateRead count{};
  NestedAggregateWorkspace tile_low{};
  NestedAggregateWorkspace tile_status{};
  NestedScalarExpr action_expr{};
  NestedScalarExpr fold_expr{};
  BackendPublish publication{};
  std::uint32_t publication_index{NoNode};
  NestedAggregateFailureProjection failure{};
  NestedAggregateProfileProjection profile{};
  const char *reason{"compute_pipeline_nested_aggregate_ineligible"};

  [[nodiscard]] constexpr bool ready() const noexcept {
    return state == NestedAggregateState::Ready;
  }

  [[nodiscard]] constexpr bool invalid() const noexcept {
    return state == NestedAggregateState::Invalid;
  }
};

// Eligibility depends only on graph semantics and resident identity. It never
// branches on tile count, iteration count, device family, or measured timing.
[[nodiscard]] MapRecurrence
BuildMapRecurrence(std::span<const BackendBatchEntry> entries,
                   std::span<const std::uint8_t> barriers);

// Nested Action barriers include the dependency entering the first Action.
// That boundary is outside the recurrence; every boundary within the Action
// subrange must still be present and ordered.
[[nodiscard]] MapRecurrence
BuildNestedMapRecurrence(std::span<const BackendBatchEntry> entries,
                         std::span<const std::uint8_t> barriers,
                         const NestedTemplateGeometry &geometry);

[[nodiscard]] NestedAggregate
BuildNestedAggregate(std::span<const BackendBatchEntry> templates,
                     std::span<const std::uint8_t> barriers,
                     std::span<const BackendPublish> publications,
                     std::size_t first);

} // namespace rund::node::accel::detail
