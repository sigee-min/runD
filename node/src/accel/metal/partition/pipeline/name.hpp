#pragma once

#include "../local.hpp"
#include "wide.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline constexpr const char* kPartitionClassifyKey = "partition.classify.u32";
inline constexpr const char* kPartitionClassifyFunction =
    "rund_compute_partition_classify_u32";
inline constexpr const char* kPartitionScatterU32Key = "partition.scatter.u32";
inline constexpr const char* kPartitionScatterU32Function =
    "rund_compute_partition_scatter_u32";
inline constexpr const char* kPartitionScatterU64Key = "partition.scatter.u64";
inline constexpr const char* kPartitionScatterU64Function =
    "rund_compute_partition_scatter_u64";
#endif

}  // namespace rund::node::accel::detail
