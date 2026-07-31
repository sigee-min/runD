#pragma once

#include "../../segmented/metal.hpp"
#include "../../segmented/model.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"

#include <memory>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kSegmentedScanWidth = 256u;

struct MetalSegmentedScanEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::SegmentedScanPlan plan{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult heads{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer offsets{};
  MetalRuntimeBuffer first_heads{};
  MetalRuntimeBuffer status{};
  std::shared_ptr<void> block{};
  std::shared_ptr<void> prefix{};
  std::shared_ptr<void> offset{};
};

void DestroyMetalSegmentedScanEncodeResources(void *raw);
[[nodiscard]] std::string MetalSegmentedScanSource();
[[nodiscard]] bool CompileMetalSegmentedScanPipelines(
    MetalAdapter &adapter, rund::kernel::SegmentedScanElement element,
    rund::kernel::ComputeDomain domain, std::shared_ptr<void> &block,
    std::shared_ptr<void> &prefix, std::shared_ptr<void> &offset);

} // namespace rund::node::accel::detail
