#pragma once

#include "../internal.hpp"

#include <kernel/program/compute/lowering/metal/source.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace rund::node::accel::detail::device_vsm_graph_pointwise::metal {

[[nodiscard]] bool append_map_binding(std::string &, bool);

[[nodiscard]] bool append_ring_bindings(std::string &, std::size_t,
                                        std::size_t);

[[nodiscard]] bool
append_fixed_helpers(std::string &,
                     std::span<const DeviceVsmGraphPointwiseStage>,
                     const rund::kernel::ArtifactKey &);

[[nodiscard]] bool
append_chained_value(std::string &, const rund::kernel::ArtifactKey &,
                     const rund::kernel::compute_lowering_detail::ParsedIR &,
                     const DeviceVsmGraphPointwiseStageTopology &,
                     std::span<const std::string>, std::span<const std::string>,
                     std::string &);

[[nodiscard]] bool append_page_ring(
    std::string &, const rund::kernel::ArtifactKey &,
    std::span<const DeviceVsmGraphPointwiseStage>,
    const DeviceVsmGraphPointwiseTopology &, const DeviceVsmPageMap &,
    const DeviceVsmGraphWavefrontProof &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    bool);

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::metal
