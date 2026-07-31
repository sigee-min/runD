#pragma once

#include "bucket.hpp"

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kVulkanSortThreadCount = 256u;
inline constexpr rund::kernel::u32 kVulkanSortItemsPerThread = 2u;
inline constexpr rund::kernel::u32 kVulkanSortBlockSize =
    kVulkanSortThreadCount * kVulkanSortItemsPerThread;
inline constexpr rund::kernel::u32 kVulkanSortRankGroupSize = 8u;
inline constexpr rund::kernel::u32 kVulkanSortRankGroupCount =
    kVulkanSortBlockSize / kVulkanSortRankGroupSize;
inline constexpr rund::kernel::u32 kVulkanSortGroupsPerWord = 8u;
inline constexpr rund::kernel::u32 kVulkanSortPackedWordCount =
    kVulkanSortRankGroupCount / kVulkanSortGroupsPerWord;
inline constexpr rund::kernel::u64 kVulkanSortSharedBytes =
    kVulkanSortBlockSize * sizeof(rund::kernel::u32) +
    kSortBucketCount * kVulkanSortPackedWordCount * sizeof(rund::kernel::u32);
static_assert(kVulkanSortRankGroupCount % kVulkanSortGroupsPerWord == 0u);
static_assert(kVulkanSortRankGroupSize <= 15u);
static_assert(kVulkanSortSharedBytes == 10u * 1024u);

} // namespace rund::node::accel::detail
