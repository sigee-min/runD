#pragma once

#include "../../adapter.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] std::shared_ptr<void>
CompileMetalMapArtifactPipeline(MetalAdapter &adapter,
                                const rund::kernel::LoweringArtifact &artifact);
#endif

} // namespace rund::node::accel::detail
