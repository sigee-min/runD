#pragma once

#include "../scan.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <string>

namespace rund::node::accel::detail::device_vsm_scan_source {

[[nodiscard]] std::string metal_source_u32(const rund::kernel::ArtifactKey &,
                                           rund::kernel::ScanOp,
                                           DeviceVsmScanMap);
[[nodiscard]] std::string metal_source_u64(const rund::kernel::ArtifactKey &,
                                           rund::kernel::ScanOp,
                                           DeviceVsmScanMap);
[[nodiscard]] std::string vulkan_source_u32(const rund::kernel::ArtifactKey &,
                                            rund::kernel::ScanOp,
                                            DeviceVsmScanMap);
[[nodiscard]] std::string vulkan_source_u64(const rund::kernel::ArtifactKey &,
                                            rund::kernel::ScanOp,
                                            DeviceVsmScanMap);

} // namespace rund::node::accel::detail::device_vsm_scan_source
