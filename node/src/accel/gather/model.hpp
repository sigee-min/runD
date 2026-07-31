#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

struct GatherParams final {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 source_count = 0u;
  rund::kernel::u32 count_source = 0u;
  rund::kernel::u32 reserved = 0u;
};

static_assert(std::is_standard_layout_v<GatherParams>);
static_assert(std::is_trivially_copyable_v<GatherParams>);
static_assert(sizeof(GatherParams) == 24u);
static_assert(alignof(GatherParams) == alignof(rund::kernel::u64));
static_assert(offsetof(GatherParams, element_count) == 0u);
static_assert(offsetof(GatherParams, source_count) == 8u);
static_assert(offsetof(GatherParams, count_source) == 16u);
static_assert(offsetof(GatherParams, reserved) == 20u);
static_assert(sizeof(decltype(GatherParams::element_count)) == 8u);
static_assert(sizeof(decltype(GatherParams::source_count)) == 8u);
static_assert(sizeof(decltype(GatherParams::count_source)) == 4u);
static_assert(sizeof(decltype(GatherParams::reserved)) == 4u);

} // namespace rund::node::accel::detail
