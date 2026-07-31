#pragma once

#include "../../../segmented/reduce/metal.hpp"
#include "../../buffer/owner.hpp"
#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../partition/local.hpp"
#include "../../reduce/local.hpp"
#include "../../runtime/map/resources.hpp"
#include "../../scan/kernel/local.hpp"
#include "../../scatter/local.hpp"
#include "../../segmented/local.hpp"
#include "../../sort/local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

template <class Resources>
[[nodiscard]] inline const Resources *
MetalResources(const std::shared_ptr<void> &resources) noexcept {
  return static_cast<const Resources *>(resources.get());
}

[[nodiscard]] inline PreparedMemory
MetalMapStepMemory(const std::shared_ptr<void> &resources,
                   const std::uint64_t budget) {
  const auto *const map = MetalResources<MetalMapEncodeResources>(resources);
  return map == nullptr ? PreparedMemory{}
                        : MetalBuffersMemory(
                              budget, map->param, map->control_args,
                              map->control_params, map->control_status);
}

[[nodiscard]] inline PreparedMemory
MetalScanStepMemory(const std::shared_ptr<void> &resources,
                    const std::uint64_t budget) {
  const auto *const scan = MetalResources<MetalScanEncodeResources>(resources);
  return scan == nullptr
             ? PreparedMemory{}
             : MetalBuffersMemory(budget, scan->totals, scan->status);
}

[[nodiscard]] inline PreparedMemory
MetalSegmentedStepMemory(const std::shared_ptr<void> &resources,
                         const std::uint64_t budget) {
  const auto *const scan =
      MetalResources<MetalSegmentedScanEncodeResources>(resources);
  return scan == nullptr ? PreparedMemory{}
                         : MetalBuffersMemory(budget, scan->offsets,
                                              scan->first_heads, scan->status);
}

[[nodiscard]] inline PreparedMemory
MetalSortStepMemory(const std::shared_ptr<void> &resources,
                    const std::uint64_t budget) {
  const auto *const sort = MetalResources<MetalSortEncodeResources>(resources);
  if (sort == nullptr) {
    return {};
  }
  PreparedMemory memory = MetalBuffersMemory(
      budget, sort->temp_keys, sort->temp_values, sort->block_counts,
      sort->block_offsets, sort->bucket_offsets, sort->dispatch_args,
      sort->status);
  return memory;
}

[[nodiscard]] inline PreparedMemory
MetalCompactStepMemory(const std::shared_ptr<void> &resources,
                       const std::uint64_t budget) {
  const auto *const compact =
      MetalResources<MetalCompactEncodeResources>(resources);
  return compact == nullptr
             ? PreparedMemory{}
             : MetalBuffersMemory(budget, compact->offsets, compact->flag_bits,
                                  compact->block_counts, compact->block_offsets,
                                  compact->scan_totals, compact->scan_status,
                                  compact->status);
}

[[nodiscard]] inline PreparedMemory
MetalGatherStepMemory(const std::shared_ptr<void> &resources,
                      const std::uint64_t budget) {
  const auto *const gather =
      MetalResources<MetalGatherEncodeResources>(resources);
  return gather == nullptr ? PreparedMemory{}
                           : MetalBufferMemory(gather->status, budget);
}

[[nodiscard]] inline PreparedMemory
MetalHistogramStepMemory(const std::shared_ptr<void> &resources,
                         const std::uint64_t budget) {
  const auto *const histogram =
      MetalResources<MetalHistogramEncodeResources>(resources);
  return histogram == nullptr ? PreparedMemory{}
                              : MetalBufferMemory(histogram->status, budget);
}

[[nodiscard]] inline PreparedMemory
MetalPartitionStepMemory(const std::shared_ptr<void> &resources,
                         const std::uint64_t budget) {
  const auto *const partition =
      MetalResources<MetalPartitionEncodeResources>(resources);
  return partition == nullptr
             ? PreparedMemory{}
             : MetalBuffersMemory(
                   budget, partition->false_bits, partition->false_offsets,
                   partition->false_totals, partition->false_status);
}

[[nodiscard]] inline PreparedMemory
MetalReduceStepMemory(const std::shared_ptr<void> &resources,
                      const std::uint64_t budget) {
  const auto *const reduce =
      MetalResources<MetalReduceEncodeResources>(resources);
  return reduce == nullptr
             ? PreparedMemory{}
             : MetalBuffersMemory(budget, reduce->partial, reduce->status);
}

[[nodiscard]] inline PreparedMemory
MetalScatterStepMemory(const std::shared_ptr<void> &resources,
                       const std::uint64_t budget) {
  const auto *const scatter =
      MetalResources<MetalScatterEncodeResources>(resources);
  return scatter == nullptr ? PreparedMemory{}
                            : MetalBufferMemory(scatter->status, budget);
}

#endif

} // namespace rund::node::accel::detail
