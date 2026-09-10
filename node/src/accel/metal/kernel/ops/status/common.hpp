#pragma once

#include "../model.hpp"

#include <rund/compute/reason.hpp>

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace metal_pipeline_status {

[[nodiscard]] constexpr std::uint32_t
reason(const rund::compute::Reason value) noexcept {
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] inline MetalPipelineStatusBinding
binding(const MetalRuntimeBuffer &buffer,
        const MetalPipelineStatusEncoding encoding,
        const std::array<std::uint32_t, 4u> reasons,
        const std::uint32_t reset = 0u, const std::uint64_t limit = 0u,
        const std::uint32_t observed = 0u) noexcept {
  return MetalPipelineStatusBinding{
      .buffer = buffer.buffer.get(),
      .offset = buffer.offset,
      .bytes = buffer.bytes,
      .limit = limit,
      .reasons = reasons,
      .reset = reset,
      .observed = observed,
      .encoding = encoding,
  };
}

[[nodiscard]] inline bool append(MetalPipelineStatusBindings &out,
                                 const MetalPipelineStatusBinding value) {
  if (value.buffer == nullptr || value.bytes == 0u ||
      (value.bytes & (sizeof(std::uint32_t) - 1u)) != 0u ||
      value.observed >= value.bytes / sizeof(std::uint32_t) ||
      out.size >= out.values.size()) {
    return false;
  }
  out.values[out.size++] = value;
  return true;
}

} // namespace metal_pipeline_status
#endif
} // namespace rund::node::accel::detail
