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
#include "../scan/pipeline.hpp"
#include <cstdint>
#include <kernel/program/compute/scan/plan.hpp>
#include <memory>
#include <optional>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kPartitionThreadgroupSize = 256u;

[[nodiscard]] constexpr rund::kernel::ScanDesc
MetalPartitionScanDesc(const rund::kernel::PartitionPlan &plan) noexcept {
  return rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = plan.element_count,
      .block_size = block::MetalPartition,
  };
}

[[nodiscard]] constexpr rund::kernel::ScanPlan
MetalPartitionScanPlan(const rund::kernel::PartitionPlan &plan) noexcept {
  return plan.ok ? rund::kernel::PlanScan(MetalPartitionScanDesc(plan))
                 : rund::kernel::ScanPlan{};
}

[[nodiscard]] constexpr RangePrefixExec
MetalPartitionScanExecution(const rund::kernel::PartitionPlan &plan) noexcept {
  return PlanScanPrefixExecution(MetalPartitionScanPlan(plan));
}

[[nodiscard]] constexpr std::size_t
MetalPartitionPipelineCount(const RangePrefixExec &execution) noexcept {
  const std::size_t scan_count = MetalScanPipelineCount(execution);
  return scan_count == 0u ? 0u : 2u + scan_count;
}

struct MetalPartitionPipelines {
  std::shared_ptr<void> classify{};
  std::shared_ptr<void> scatter{};
};

struct MetalPartitionEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::PartitionPlan plan{};
  rund::kernel::ScanDesc scan_desc{};
  rund::kernel::ScanPlan scan_plan{};
  std::optional<RangePrefixExec> scan_execution{};
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
[[nodiscard]] bool
MetalPartitionSourceUpperBytes(std::uint64_t &upper) noexcept;
[[nodiscard]] bool CompileMetalPartitionPipelines(MetalAdapter &adapter,
                                                  rund::kernel::u64 flag_bytes,
                                                  rund::kernel::u64 value_bytes,
                                                  MetalPartitionPipelines &out);

} // namespace rund::node::accel::detail
