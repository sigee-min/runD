#pragma once

#include "model.hpp"

#include <accel/check.hpp>

#include <memory>
#include <span>

namespace rund::node::accel::detail {
struct BackendBatchEntry;
}

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] rund::AccelCheck
prepare(MetalAdapter &adapter, std::span<const BackendBatchEntry> entries,
        std::shared_ptr<void> &owner, Workspace *&workspace, Maps &maps);
#endif

} // namespace rund::node::accel::detail::metalbatch
