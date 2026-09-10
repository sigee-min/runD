#pragma once

#include "../internal.hpp"

#include <string>

namespace rund::node::accel::detail::device_vsm_window_source::vulkan {

void append_update(std::string &, const RangeExec &);
[[nodiscard]] bool append_saturating_algebra(std::string &);
void append_direct(std::string &, const RangeExec &, const char *,
                   const DeviceVsmWindowFusion &);
void append_shared(std::string &, const RangeExec &, const char *,
                   const DeviceVsmWindowFusion &);
void append_counters(std::string &);

[[nodiscard]] bool emit_ring(const RangeExec &,
                             const rund::kernel::ArtifactKey &,
                             const DeviceVsmWindowFusion &,
                             const DeviceVsmWindowRingPlan &,
                             const DeviceVsmWindowMapSources &,
                             std::string &);

} // namespace rund::node::accel::detail::device_vsm_window_source::vulkan
