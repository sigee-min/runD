#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

struct PartitionParams final {
  rund::kernel::u64 element_count = 0u;
};

static_assert(std::is_standard_layout_v<PartitionParams>);
static_assert(std::is_trivially_copyable_v<PartitionParams>);
static_assert(sizeof(PartitionParams) == 8u);
static_assert(alignof(PartitionParams) == alignof(rund::kernel::u64));
static_assert(offsetof(PartitionParams, element_count) == 0u);
static_assert(sizeof(decltype(PartitionParams::element_count)) == 8u);

} // namespace rund::node::accel::detail
