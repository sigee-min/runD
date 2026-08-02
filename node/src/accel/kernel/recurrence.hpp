#pragma once

#include "backend/run.hpp"
#include "prepared/template_registry.hpp"

#include <kernel/program/compute/limit.hpp>

#include <array>
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
