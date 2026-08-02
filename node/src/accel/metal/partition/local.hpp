#pragma once

#include "../../partition.hpp"
#include "../../partition/model.hpp"
#include "../../primitive/block.hpp"
#include "../../scan/metal.hpp"
#include "../../scan/shape.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kPartitionThreadgroupSize = 256u;
struct MetalPartitionPipelines {
  std::shared_ptr<void> classify{};
  std::shared_ptr<void> scatter{};
};

struct MetalPartitionEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::PartitionPlan plan{};
  rund::kernel::ScanDesc scan_desc{};
  rund::kernel::ScanPlan scan_plan{};
  MetalResidentBufferResult flags{};
  MetalResidentBufferResult values{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer false_bits{};
  MetalRuntimeBuffer false_offsets{};
  MetalRuntimeBuffer false_totals{};
  MetalRuntimeBuffer false_status{};
  MetalPartitionPipelines pipelines{};
  std::shared_ptr<void> scan_block{};
  std::shared_ptr<void> scan_prefix{};
  std::shared_ptr<void> scan_offset{};
};

[[nodiscard]] inline bool MetalPartitionBuffersReady(
    const MetalPartitionEncodeResources &partition) noexcept {
  return partition.false_bits.buffer != nullptr &&
         partition.false_offsets.buffer != nullptr &&
         partition.false_totals.buffer != nullptr &&
         partition.false_status.buffer != nullptr;
}

void DestroyMetalPartitionEncodeResources(void *raw);
[[nodiscard]] std::string MetalPartitionSource();
[[nodiscard]] bool MetalPartitionSourceUpperBytes(
    std::uint64_t &upper) noexcept;
[[nodiscard]] bool CompileMetalPartitionPipelines(MetalAdapter &adapter,
                                                  rund::kernel::u64 flag_bytes,
                                                  rund::kernel::u64 value_bytes,
                                                  MetalPartitionPipelines &out);

} // namespace rund::node::accel::detail
