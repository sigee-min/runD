#pragma once

#include "../state.hpp"

#include "../host_ring.hpp"

#include <optional>
#include <span>

namespace rund::compute::detail::virtual_prepare_detail {

// Preparation freezes these two immutable records before any physical Pool
// or Pipeline is published. Runtime routes consume the records verbatim.
[[nodiscard]] std::optional<VirtualGeometry>
classify_geometry(const ProgramState &) noexcept;

[[nodiscard]] VirtualWindowPreflight preflight_window(
    const ProgramState &, const VirtualGeometry &, const VirtualBufferState &,
    const VirtualBufferState &, std::uint64_t page_count,
    ResidencyConfig) noexcept;

[[nodiscard]] Status
validate_pipeline_capability(const DeviceState &) noexcept;

} // namespace rund::compute::detail::virtual_prepare_detail
