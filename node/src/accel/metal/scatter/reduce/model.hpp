#pragma once

#include "../../../scatter.hpp"

#include "../../buffer/owner.hpp"
#include "../../buffer/resident/batch.hpp"
#include "../../state.hpp"

#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalScatterReduceResources final {
  MetalAdapter *adapter = nullptr;
  rund::kernel::ScatterReducePlan plan{};
  MetalResidentBufferResult values{};
  MetalResidentBufferResult indices{};
  MetalResidentBufferResult count{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer status{};
  MetalRuntimeBuffer indirect{};
  MetalRuntimeBuffer counts{};
  std::shared_ptr<void> control_pipeline{};
  std::shared_ptr<void> init_pipeline{};
  std::shared_ptr<void> fold_pipeline{};
};

[[nodiscard]] std::string
MetalScatterReduceKey(const rund::kernel::ScatterReducePlan &plan);
[[nodiscard]] std::string
MetalScatterReduceSource(const rund::kernel::ScatterReducePlan &plan);
[[nodiscard]] bool MetalScatterReduceSourceUpperBytes(
    const rund::kernel::ScatterReducePlan &plan,
    std::uint64_t &upper) noexcept;

[[nodiscard]] bool AcquireMetalScatterReducePipelines(
    MetalAdapter &adapter, const rund::kernel::ScatterReducePlan &plan,
    std::shared_ptr<void> &control, std::shared_ptr<void> &init,
    std::shared_ptr<void> &fold);

void DestroyMetalScatterReduce(void *raw);

#endif

} // namespace rund::node::accel::detail
