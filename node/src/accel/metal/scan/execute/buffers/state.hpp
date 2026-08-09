#pragma once

#include "../../../command/run.hpp"
#include "../../local.hpp"

#include <optional>
#include <utility>

namespace rund::node::accel::detail {

struct MetalScanDirectBuffers {
  std::optional<RangePrefixExec> prefix_execution{};
  MetalRuntimeBuffer totals{};
  MetalRuntimeBuffer status{};
};

inline void ReleaseMetalScanDirectBuffers(MetalAdapter &adapter,
                                          MetalScanDirectBuffers &buffers) {
  ReleaseMetalBuffer(adapter, std::move(buffers.totals));
  ReleaseMetalBuffer(adapter, std::move(buffers.status));
}

} // namespace rund::node::accel::detail
