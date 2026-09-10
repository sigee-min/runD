#pragma once

#include "../internal.hpp"

#include <kernel/program/compute/lowering/vulkan/source.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan {

using MapRows = std::array<std::size_t, DeviceVsmGraphStageInputCapacity>;

[[nodiscard]] bool install_map_binding(std::string &, bool);

[[nodiscard]] bool append_storage_bindings(std::string &, std::size_t,
                                           std::size_t);

[[nodiscard]] MapRows collect_map_rows(const DeviceVsmPageMap &);

[[nodiscard]] bool
append_stage_fixed_helpers(std::string &,
                           std::span<const DeviceVsmGraphPointwiseStage>,
                           const rund::kernel::ArtifactKey &);

[[nodiscard]] bool
append_chained_value(std::string &, const rund::kernel::ArtifactKey &,
                     const rund::kernel::compute_lowering_detail::ParsedIR &,
                     const DeviceVsmGraphPointwiseStageTopology &,
                     std::span<const std::string>, std::span<const std::string>,
                     std::string &);

[[nodiscard]] bool append_body(
    std::string &, const rund::kernel::ArtifactKey &,
    std::span<const DeviceVsmGraphPointwiseStage>,
    const DeviceVsmGraphPointwiseTopology &, const DeviceVsmPageMap &,
    const DeviceVsmGraphWavefrontProof &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    std::span<const std::string>, const MapRows &);

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan
