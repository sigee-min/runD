#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool ScatterStagedOutput(
    MetalAdapter& adapter,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const StagedProof& staged,
    const MetalRuntimeBuffer& output,
    const std::size_t output_size) {
  rund::kernel::u64 copied = 0u;
  if (!ScatterOutputBytes(bindings, window, MetalBufferContents(output),
                          output_size, staged, copied)) {
    return false;
  }
  RecordMetalDeviceToHostBytes(adapter, copied);
  return true;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
