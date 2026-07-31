#pragma once

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline constexpr const char* kPartitionClassifyU64Key =
    "partition.classify.u64";
inline constexpr const char* kPartitionClassifyU64Function =
    "rund_compute_partition_classify_u64";
inline constexpr const char* kPartitionScatterF64V32Key =
    "partition.scatter.f64.v32";
inline constexpr const char* kPartitionScatterF64V32Function =
    "rund_compute_partition_scatter_f64_v32";
inline constexpr const char* kPartitionScatterF64V64Key =
    "partition.scatter.f64.v64";
inline constexpr const char* kPartitionScatterF64V64Function =
    "rund_compute_partition_scatter_f64_v64";
#endif

} // namespace rund::node::accel::detail
