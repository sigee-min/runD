#pragma once

#include "model.hpp"

namespace rund::node {

[[nodiscard]] ReactorProbeResult
ReactorProbeNow(ReactorPlatform &platform, std::vector<BatchIoReady> &scratch,
                const ReactorRequest &request) noexcept;

} // namespace rund::node
