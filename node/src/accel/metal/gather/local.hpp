#pragma once

#include "../../gather.hpp"
#include "../../gather/model.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include <memory>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kGatherThreadgroupSize = 256u;

struct MetalGatherEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::GatherPlan plan{};
  MetalResidentBufferResult values{};
  MetalResidentBufferResult indices{};
  MetalResidentBufferResult logical_count{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer status{};
  MetalRuntimeBuffer indirect{};
  std::shared_ptr<void> control_pipeline{};
  std::shared_ptr<void> gather_pipeline{};
};

void DestroyMetalGatherEncodeResources(void *raw);
[[nodiscard]] std::string MetalGatherSource();
[[nodiscard]] std::uint64_t MetalGatherSourceUpperBytes() noexcept;
[[nodiscard]] bool CompileMetalGatherPipelines(
    MetalAdapter &adapter, rund::kernel::GatherElement element,
    std::shared_ptr<void> &control, std::shared_ptr<void> &gather);

} // namespace rund::node::accel::detail
