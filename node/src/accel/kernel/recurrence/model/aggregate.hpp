#pragma once

#include "nested_geometry.hpp"

namespace rund::node::accel::detail {

struct NestedAggregateRead final {
  BackendRead read{};
  std::uint64_t logical_count{};
  std::uint64_t element_bytes{};
};

struct NestedAggregateWorkspace final {
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
};

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

[[nodiscard]] MapRecurrence
BuildMapRecurrence(std::span<const BackendBatchEntry> entries,
                   std::span<const std::uint8_t> barriers);

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
