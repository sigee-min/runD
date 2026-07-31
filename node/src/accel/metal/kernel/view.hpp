#pragma once

#include "../../kernel/backend/run.hpp"
#include "../../kernel/memory.hpp"
#include "../../kernel/preparation.hpp"
#include "../resident/model.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalViewTransfer final {
  std::uint64_t binding{};
  MetalResidentBufferResult external{};
  MetalResidentBufferResult dense{};
  std::uint64_t count{};
  std::uint64_t element_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t stride_bytes{};
  bool input{};
  bool planned{};
};

struct MetalViewLowering final {
  RunBinds binds{};
  BoundStep step{};
  std::vector<MetalViewTransfer> transfers{};
  std::vector<std::uint32_t> transfer_by_binding{};
  std::shared_ptr<void> gather_pipeline{};
  std::shared_ptr<void> scatter_pipeline{};
  bool has_input{};
};

[[nodiscard]] rund::AccelCheck PrepareMetalViewLowering(
    const rund::AccelDevice &pick, const BoundStep &source,
    KernelPreparationMode mode, const KernelViewLayout *views,
    const RunBinds *view_binds, std::shared_ptr<MetalViewLowering> &out);

[[nodiscard]] rund::AccelCheck
EncodeMetalViewInputs(const std::shared_ptr<MetalViewLowering> &view,
                      id<MTLComputeCommandEncoder> encoder);

[[nodiscard]] rund::AccelCheck
EncodeMetalViewOutputs(const std::shared_ptr<MetalViewLowering> &view,
                       id<MTLComputeCommandEncoder> encoder);

[[nodiscard]] PreparedMemory
MetalViewMemory(const std::shared_ptr<MetalViewLowering> &view,
                std::uint64_t budget, std::uint64_t &traffic) noexcept;

[[nodiscard]] std::uint64_t
MetalViewDispatchCount(const std::shared_ptr<MetalViewLowering> &view) noexcept;

#endif

} // namespace rund::node::accel::detail
