#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::select_missing(
    PrefetchLane &lane, const residency::PoolPhysicalOwner &input_owner,
    const bool speculative) noexcept {
  std::array<residency::CacheKey, PipelineLeafCapacity> input_keys{};
  std::array<std::uint8_t, PipelineLeafCapacity> device_resident{};
  std::array<std::uint8_t, PipelineLeafCapacity> bank_resident{};
  for (std::size_t index = 0u; index < lane.count; ++index) {
    if (!residency::project_graph_cache_key(
            lane.materialization, lane.sources[index].key, input_keys[index])) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }

  // An immediately consumed read-only probe is safe for selection. A
  // speculative e+D observation has no Authority pin, so it retains the Host
  // receipt as its conservative fallback until begin_graph_epoch authenticates
  // the later Device hit.
  if (!speculative) {
    for (const residency::FrameRegion cache : input_owner.cache_regions) {
      bank_resident.fill(0u);
      if (!authority_.probe(
              std::span<const residency::CacheKey>{input_keys.data(),
                                                   lane.count},
              std::span<std::uint8_t>{bank_resident.data(), lane.count},
              cache.first, cache.count)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      for (std::size_t index = 0u; index < lane.count; ++index) {
        device_resident[index] = static_cast<std::uint8_t>(
            device_resident[index] != 0u || bank_resident[index] != 0u);
      }
    }
  }

  std::size_t missing_count = 0u;
  for (std::size_t index = 0u; index < lane.count; ++index) {
    if (device_resident[index] == 0u) {
      lane.sources[missing_count++] = lane.sources[index];
    }
  }
  lane.count = missing_count;
  lane.device_only = missing_count == 0u;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
