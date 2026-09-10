#pragma once

#include "source/window.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

// Rewrites one canonical pointwise Map into a fixed-W device page controller.
// Q and page sizes remain runtime values, so retained source storage is
// independent of the page count.
[[nodiscard]] bool TransformDeviceVsmSource(rund::kernel::LoweringArtifact &,
                                            std::uint64_t input_count,
                                            std::uint64_t output_count);

} // namespace rund::node::accel::detail
