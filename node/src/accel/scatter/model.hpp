#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kScatterStatusOk = 0xffffffffu;
inline constexpr rund::kernel::u32 kScatterReasonOutOfRange = 0u;
inline constexpr rund::kernel::u32 kScatterReasonDuplicate = 1u;

struct ScatterParams final {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 output_count = 0u;
};

static_assert(std::is_standard_layout_v<ScatterParams>);
static_assert(std::is_trivially_copyable_v<ScatterParams>);
static_assert(sizeof(ScatterParams) == 16u);
static_assert(alignof(ScatterParams) == alignof(rund::kernel::u64));
static_assert(offsetof(ScatterParams, element_count) == 0u);
static_assert(offsetof(ScatterParams, output_count) == 8u);
static_assert(sizeof(decltype(ScatterParams::element_count)) == 8u);
static_assert(sizeof(decltype(ScatterParams::output_count)) == 8u);

} // namespace rund::node::accel::detail
