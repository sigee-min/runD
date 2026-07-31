#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

struct SegmentedScanParams final {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 block_size = 0u;
  rund::kernel::u64 block_count = 0u;
  rund::kernel::u32 inclusive = 0u;
  rund::kernel::u32 reserved = 0u;
};

static_assert(std::is_standard_layout_v<SegmentedScanParams>);
static_assert(std::is_trivially_copyable_v<SegmentedScanParams>);
static_assert(sizeof(SegmentedScanParams) == 32u);
static_assert(alignof(SegmentedScanParams) == alignof(rund::kernel::u64));
static_assert(offsetof(SegmentedScanParams, element_count) == 0u);
static_assert(offsetof(SegmentedScanParams, block_size) == 8u);
static_assert(offsetof(SegmentedScanParams, block_count) == 16u);
static_assert(offsetof(SegmentedScanParams, inclusive) == 24u);
static_assert(offsetof(SegmentedScanParams, reserved) == 28u);
static_assert(sizeof(decltype(SegmentedScanParams::element_count)) == 8u);
static_assert(sizeof(decltype(SegmentedScanParams::block_size)) == 8u);
static_assert(sizeof(decltype(SegmentedScanParams::block_count)) == 8u);
static_assert(sizeof(decltype(SegmentedScanParams::inclusive)) == 4u);
static_assert(sizeof(decltype(SegmentedScanParams::reserved)) == 4u);

} // namespace rund::node::accel::detail
