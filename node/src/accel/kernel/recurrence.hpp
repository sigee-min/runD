#pragma once

#include "backend/run.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
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
  std::vector<rund::kernel::ResidentBufferRef> outputs{};
  std::vector<std::shared_ptr<void>> handles{};
  std::vector<std::uint64_t> pitch_bytes{};

  [[nodiscard]] rund::kernel::ResidentBindingRange range() const noexcept {
    if (outputs.empty() || outputs.size() != handles.size() ||
        outputs.size() != pitch_bytes.size()) {
      return {};
    }
    return rund::kernel::ResidentBindingRange{
        .refs = outputs.data(),
        .handles = handles.data(),
        .storage_count = static_cast<std::uint64_t>(outputs.size()),
        .count = static_cast<std::uint64_t>(outputs.size()),
    };
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
  rund::kernel::LoweringArtifact artifact{};
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

struct NestedTemplateSpan final {
  std::uint32_t first{};
  std::uint32_t count{};

  [[nodiscard]] constexpr std::uint64_t end() const noexcept {
    return static_cast<std::uint64_t>(first) + count;
  }
};

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

// Exact projection from one aggregate command back to the authored recurrence.
// A backend may select the aggregate for profiled execution only when it can
// project every authored row and the sole physical owner from these counts.
struct NestedAggregateProfileProjection final {
  std::uint64_t authored_seed_occurrences{};
  std::uint64_t authored_action_occurrences{};
  std::uint64_t authored_fold_occurrences{};
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
// Template spans remain the binding/resource authority. Backends consume the
// normalized opcodes and must not rediscover them from generated source.
struct NestedAggregate final {
  NestedAggregateState state{NestedAggregateState::Ineligible};
  NestedAggregateKind kind{NestedAggregateKind::WindowIndexedReduceSumU32};
  NestedTemplateSpan seed{};
  NestedTemplateSpan action{};
  NestedTemplateSpan fold{};
  BackendWindow window{};
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

  [[nodiscard]] constexpr std::uint64_t
  authored_occurrences(const std::size_t template_index) const noexcept {
    if (template_index >= seed.first && template_index < seed.end()) {
      return 1u;
    }
    if (template_index >= action.first && template_index < action.end()) {
      return window.outer_bound;
    }
    if (template_index < fold.first || template_index >= fold.end()) {
      return 0u;
    }
    const std::size_t route = template_index - fold.first;
    if (route == 0u) {
      return 1u;
    }
    if (route == 1u) {
      return window.outer_bound / 2u;
    }
    return window.outer_bound == 0u ? 0u : (window.outer_bound - 1u) / 2u;
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
                         std::span<const std::uint8_t> barriers);

[[nodiscard]] NestedAggregate
BuildNestedAggregate(std::span<const BackendBatchEntry> templates,
                     std::span<const std::uint8_t> barriers,
                     std::span<const BackendPublish> publications,
                     std::size_t first);

} // namespace rund::node::accel::detail
