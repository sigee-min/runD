#pragma once

#include "../graph_map_reduce.hpp"
#include "../typed_map/internal.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/layout.hpp>

#include <string>
#include <vector>

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

[[nodiscard]] bool
validate(const rund::kernel::LoweringArtifact &,
         const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
         const MapSemantic &, const rund::kernel::ReducePlan &) noexcept;
[[nodiscard]] std::string
metal_source(const rund::kernel::ArtifactKey &,
             const rund::kernel::compute_lowering_detail::ParsedIR &,
             rund::kernel::ReduceOp, const DeviceVsmGraphWavefrontProof &);
[[nodiscard]] std::string
metal_additive_source(const rund::kernel::ArtifactKey &,
                      const rund::kernel::compute_lowering_detail::ParsedIR &,
                      rund::kernel::ReduceOp,
                      const DeviceVsmGraphWavefrontProof &);
[[nodiscard]] std::string
metal_extreme_source(const rund::kernel::ArtifactKey &,
                     const rund::kernel::compute_lowering_detail::ParsedIR &,
                     rund::kernel::ReduceOp,
                     const DeviceVsmGraphWavefrontProof &);
[[nodiscard]] std::string
vulkan_source(const rund::kernel::ArtifactKey &,
              const rund::kernel::compute_lowering_detail::ParsedIR &,
              rund::kernel::ReduceOp, const DeviceVsmGraphWavefrontProof &);
[[nodiscard]] std::string
vulkan_additive_source(const rund::kernel::ArtifactKey &,
                       const rund::kernel::compute_lowering_detail::ParsedIR &,
                       rund::kernel::ReduceOp,
                       const DeviceVsmGraphWavefrontProof &);
[[nodiscard]] std::string
vulkan_extreme_source(const rund::kernel::ArtifactKey &,
                      const rund::kernel::compute_lowering_detail::ParsedIR &,
                      rund::kernel::ReduceOp,
                      const DeviceVsmGraphWavefrontProof &);

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
