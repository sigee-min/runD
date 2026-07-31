#pragma once

#include "state.hpp"

#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool
CollectMetalStatus(MetalKernelResources &resources, std::uint32_t declared_step,
                   std::vector<MetalPipelineStatusBindingRecord> &bindings,
                   std::vector<MetalPipelineStatusSourceMeta> &sources,
                   std::uint32_t &raw_count, std::uint32_t &entry_count);

[[nodiscard]] bool
ValidMetalReset(std::span<const MetalPipelineResetMeta> resets,
                std::uint32_t raw_count) noexcept;

[[nodiscard]] bool PrepareMetalStatus(
    MetalAdapter &adapter, bool need_status, bool need_reset, bool need_import,
    bool need_telemetry, bool profile_steps, std::shared_ptr<void> &reset,
    std::shared_ptr<void> &import, std::shared_ptr<void> &reduce,
    std::shared_ptr<void> &complete, std::shared_ptr<void> &telemetry,
    bool need_publish, std::shared_ptr<void> &publish, bool need_advance,
    std::shared_ptr<void> &advance, bool need_gate,
    std::shared_ptr<void> &gate);

#endif

} // namespace rund::node::accel::detail
