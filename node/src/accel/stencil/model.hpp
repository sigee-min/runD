#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

struct StencilParams final {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 radius = 1u;
};

static_assert(std::is_standard_layout_v<StencilParams>);
static_assert(std::is_trivially_copyable_v<StencilParams>);
static_assert(sizeof(StencilParams) == 16u);
static_assert(alignof(StencilParams) == alignof(rund::kernel::u64));
static_assert(offsetof(StencilParams, element_count) == 0u);
static_assert(offsetof(StencilParams, radius) == 8u);
static_assert(sizeof(decltype(StencilParams::element_count)) == 8u);
static_assert(sizeof(decltype(StencilParams::radius)) == 8u);

} // namespace rund::node::accel::detail
