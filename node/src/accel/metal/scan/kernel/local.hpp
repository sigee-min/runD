#pragma once

#include <accel/check.hpp>

#include "../../../scan/metal.hpp"
#include "../../../scan/shape.hpp"
#include "../../adapter.hpp"
#include "../../resident.hpp"
#include "../../state.hpp"
#include "../pipeline.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalScanEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::ScanDesc desc{};
  rund::kernel::ScanPlan plan{};
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::U32;
  rund::kernel::GraphControl control{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  MetalResidentBufferResult logical_count{};
  MetalRuntimeBuffer totals{};
  MetalRuntimeBuffer status{};
  std::shared_ptr<void> block{};
  std::shared_ptr<void> prefix{};
  std::shared_ptr<void> offset{};
};

[[nodiscard]] rund::AccelCheck EncodePreparedMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    void *input_buffer, void *output_buffer, void *totals_buffer,
    void *status_buffer, void *command_encoder,
    const std::shared_ptr<void> &block, const std::shared_ptr<void> &prefix,
    const std::shared_ptr<void> &offset, void *logical_count_buffer,
    rund::kernel::u32 count_words, rund::kernel::u64 input_offset,
    rund::kernel::u64 output_offset, rund::kernel::u64 logical_count_offset,
    rund::kernel::u64 totals_offset);
#endif
} // namespace rund::node::accel::detail
