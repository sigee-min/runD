#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

struct StencilParams final {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 radius = 1u;
  // Every RangeAggregate dispatch receives the immutable logical window plus
  // the exact stage cardinality.  The latter is never inferred from a
  // workgroup count: Prefix hierarchy and block-window stages have distinct
  // domains even when they share the same executable source.
  rund::kernel::u64 stage_element_count = 0u;
  rund::kernel::u64 stage_aux_count = 0u;
  rund::kernel::u32 stage = 0u;
  rund::kernel::u32 reserved = 0u;
};

static_assert(std::is_standard_layout_v<StencilParams>);
static_assert(std::is_trivially_copyable_v<StencilParams>);
static_assert(sizeof(StencilParams) == 40u);
static_assert(alignof(StencilParams) == alignof(rund::kernel::u64));
static_assert(offsetof(StencilParams, element_count) == 0u);
static_assert(offsetof(StencilParams, radius) == 8u);
static_assert(offsetof(StencilParams, stage_element_count) == 16u);
static_assert(offsetof(StencilParams, stage_aux_count) == 24u);
static_assert(offsetof(StencilParams, stage) == 32u);
static_assert(offsetof(StencilParams, reserved) == 36u);
static_assert(sizeof(decltype(StencilParams::element_count)) == 8u);
static_assert(sizeof(decltype(StencilParams::radius)) == 8u);

} // namespace rund::node::accel::detail
