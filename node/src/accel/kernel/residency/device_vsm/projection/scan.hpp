#pragma once

#include "../projection.hpp"

#include "../../../prepared/model.hpp"

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/scan/model.hpp>

namespace rund::node::accel::detail::device_vsm_scan_projection {

struct Authority final {
  rund::kernel::ScanPlan semantic{};
  DeviceVsmScanMap map{};
  const BoundStep *map_step{};
  rund::kernel::BindingSet map_bindings{};
  rund::kernel::ComputeApi api{rund::kernel::ComputeApi::Cpu};
};

[[nodiscard]] bool exact(const prepared::PipelineState &, Authority &,
                         const char *&reason) noexcept;
[[nodiscard]] bool same(const prepared::PipelineState &,
                        const Authority &) noexcept;

} // namespace rund::node::accel::detail::device_vsm_scan_projection
