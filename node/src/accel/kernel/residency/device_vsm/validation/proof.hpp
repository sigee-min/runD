#pragma once

#include "../graph_wavefront.hpp"
#include "../proof.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] bool device_vsm_window_ring_fusion_valid(
    const DeviceVsmWindowFusion &fusion) noexcept;

[[nodiscard]] bool
device_vsm_window_proof_valid(const DeviceVsmProof &proof) noexcept;

[[nodiscard]] bool
device_vsm_scan_proof_valid(const DeviceVsmProof &proof) noexcept;

[[nodiscard]] bool
device_vsm_reduce_proof_valid(const DeviceVsmProof &proof) noexcept;

[[nodiscard]] bool
device_vsm_topology_valid(const DeviceVsmProof &proof) noexcept;

[[nodiscard]] std::uint64_t
device_vsm_output_page_count(const DeviceVsmProof &proof) noexcept;

[[nodiscard]] bool
device_vsm_backing_valid(const rund::kernel::ResidentBufferRef &backing,
                         const std::shared_ptr<void> &handle,
                         std::uint32_t usage, std::uint32_t element_bytes,
                         std::uint64_t required_bytes) noexcept;

[[nodiscard]] bool
device_vsm_residents_valid(const DeviceVsmProof &proof) noexcept;

[[nodiscard]] bool
device_vsm_proof_valid(const DeviceVsmProof &proof) noexcept;

} // namespace rund::node::accel::detail
