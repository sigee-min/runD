#pragma once

#include "../../compact.hpp"
#include "../../primitive/block.hpp"
#include "../../scan/metal.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include <memory>
#include <string>
#include <utility>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct CompactParams {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 output_capacity = 0u;
  rund::kernel::u32 scan_block_size = 0u;
  rund::kernel::u32 reserved = 0u;
};

struct MetalCompactPipelines {
  std::shared_ptr<void> count_blocks{};
  std::shared_ptr<void> scatter{};
  std::shared_ptr<void> scatter_blocks{};
  std::shared_ptr<void> status{};
};

struct MetalCompactEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::CompactPlan plan{};
  rund::kernel::ScanDesc scan_desc{};
  rund::kernel::ScanPlan scan_plan{};
  bool block_offset_path = false;
  rund::kernel::u64 block_count = 0u;
  MetalResidentBufferResult flags{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer offsets{};
  MetalRuntimeBuffer flag_bits{};
  MetalRuntimeBuffer block_counts{};
  MetalRuntimeBuffer block_offsets{};
  MetalRuntimeBuffer scan_totals{};
  MetalRuntimeBuffer scan_status{};
  MetalRuntimeBuffer status{};
  std::shared_ptr<void> scan_block{};
  std::shared_ptr<void> scan_prefix{};
  std::shared_ptr<void> scan_offset{};
  MetalCompactPipelines pipelines{};
};

void DestroyMetalCompactEncodeResources(void *raw);
[[nodiscard]] std::string MetalCompactSource();
[[nodiscard]] bool MetalCompactSourceUpperBytes(std::uint64_t &upper) noexcept;
[[nodiscard]] bool
CompileMetalCompactPipelines(MetalAdapter &adapter,
                             MetalCompactPipelines &pipelines,
                             bool status_required, bool block_offsets);
#endif

} // namespace rund::node::accel::detail
