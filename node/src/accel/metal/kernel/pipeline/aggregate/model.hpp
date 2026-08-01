#pragma once

#include "../state.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

enum class MetalNestedScalarValue : std::uint32_t {
  TileState,
  TileCount,
  OuterState,
  Immediate,
};

struct MetalNestedScalarExpr final {
  std::uint32_t lhs{};
  std::uint32_t rhs{};
  std::uint32_t immediate{};
  std::uint32_t reserved{};
};

struct MetalNestedAggregateParams final {
  std::uint64_t queue_offset_words{};
  std::uint64_t queue_stride_words{};
  std::uint64_t domain_offset_words{};
  std::uint64_t domain_stride_words{};
  std::uint64_t count_offset_words{};
  std::uint64_t seed_offset_words{};
  std::uint64_t target_offset_words{};
  std::uint64_t tile_low_offset_words{};
  std::uint64_t tile_status_offset_words{};
  std::uint64_t queue_count{};
  std::uint64_t domain_count{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t outer_bound{};
  std::uint32_t inner_bound{};
  std::uint32_t generation_stride{};
  std::uint32_t declared_step_count{};
  std::uint32_t declared_step{};
  std::uint32_t count_overflow_reason{};
  std::uint32_t gather_reason{};
  std::uint32_t reduce_reason{};
  std::uint32_t profile_steps{};
  std::uint32_t profile_count{};
  std::uint32_t profile_seed_first{};
  std::uint32_t reserved{};
  MetalNestedScalarExpr action{};
  MetalNestedScalarExpr fold{};
};

static_assert(sizeof(MetalNestedScalarExpr) == 16u);
static_assert(sizeof(MetalNestedAggregateParams) == 176u);

struct MetalNestedAggregate final {
  std::array<std::shared_ptr<void>, 7u> buffers{};
  std::shared_ptr<void> reduce_pipeline{};
  std::shared_ptr<void> finalize_pipeline{};
  MetalNestedAggregateParams params{};
  NSUInteger threads{};
  std::uint64_t workgroup_count{};
  std::uint64_t work_item_count{};
};

#endif

} // namespace rund::node::accel::detail
