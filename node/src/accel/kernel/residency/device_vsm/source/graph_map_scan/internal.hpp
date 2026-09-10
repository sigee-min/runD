#pragma once

#include "../graph_map_scan.hpp"
#include "../typed_map/internal.hpp"

#include <string>

namespace rund::node::accel::detail::device_vsm_graph_map_scan {

[[nodiscard]] std::string
metal_source(const rund::kernel::ArtifactKey &,
             const rund::kernel::compute_lowering_detail::ParsedIR &,
             rund::kernel::ScanOp);
[[nodiscard]] std::string
vulkan_source(const rund::kernel::ArtifactKey &,
              const rund::kernel::compute_lowering_detail::ParsedIR &,
              rund::kernel::ScanOp);

} // namespace rund::node::accel::detail::device_vsm_graph_map_scan
