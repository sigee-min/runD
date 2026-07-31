#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kHistogramStatusOk = 0xffffffffu;
inline constexpr rund::kernel::u32 kHistogramReasonBinInvalid = 0u;

struct HistogramParams final {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 bin_count = 0u;
};

static_assert(std::is_standard_layout_v<HistogramParams>);
static_assert(std::is_trivially_copyable_v<HistogramParams>);
static_assert(sizeof(HistogramParams) == 16u);
static_assert(alignof(HistogramParams) == alignof(rund::kernel::u64));
static_assert(offsetof(HistogramParams, element_count) == 0u);
static_assert(offsetof(HistogramParams, bin_count) == 8u);
static_assert(sizeof(decltype(HistogramParams::element_count)) == 8u);
static_assert(sizeof(decltype(HistogramParams::bin_count)) == 8u);

} // namespace rund::node::accel::detail
