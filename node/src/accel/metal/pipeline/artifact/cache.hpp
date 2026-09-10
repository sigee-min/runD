#pragma once

#include "../../adapter.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] std::shared_ptr<void>
FindMetalMapArtifactPipeline(MetalAdapter &adapter,
                             const rund::kernel::LoweringArtifact &artifact);

[[nodiscard]] std::shared_ptr<void>
StoreMetalMapArtifactPipeline(MetalAdapter &adapter,
                              rund::kernel::LoweringArtifact artifact,
                              const std::shared_ptr<void> &pipeline);
#endif

} // namespace rund::node::accel::detail
