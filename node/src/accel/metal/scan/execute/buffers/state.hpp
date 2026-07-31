#pragma once

#include "../../../command/run.hpp"
#include "../../local.hpp"

#include <utility>

namespace rund::node::accel::detail {

struct MetalScanDirectBuffers {
  MetalRuntimeBuffer totals{};
  MetalRuntimeBuffer status{};
};

inline void ReleaseMetalScanDirectBuffers(MetalAdapter &adapter,
                                          MetalScanDirectBuffers &buffers) {
  ReleaseMetalBuffer(adapter, std::move(buffers.totals));
  ReleaseMetalBuffer(adapter, std::move(buffers.status));
}

} // namespace rund::node::accel::detail
