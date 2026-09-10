#pragma once

#include "../window.hpp"

#include "../../../../../range_aggregate/execution/projection.hpp"

#include <string>
#include <string_view>

namespace rund::node::accel::detail::device_vsm_window_source {

struct Parameters final {
  std::uint64_t input_count{};
  std::uint64_t output_count{};
  std::uint64_t window_size{};
  std::uint64_t stride{};
  std::uint64_t padding{};
  std::uint64_t stage_element_count{};
  std::uint64_t stage_aux_count{};
  std::uint32_t stage{};
  std::uint32_t reserved{};
};

static_assert(sizeof(Parameters) == DeviceVsmWindowParameterBytes);

enum class MapSlot : std::uint8_t {
  Before,
  After,
};

[[nodiscard]] std::string
entry_name(const rund::kernel::ArtifactKey &) noexcept;
[[nodiscard]] rund::kernel::ComputeDomain
executable_domain(const RangeExec &) noexcept;
[[nodiscard]] rund::kernel::ComputeApi api_for(RangeSource) noexcept;
[[nodiscard]] rund::kernel::ArtifactKey
artifact_key(RangeIdentity, RangeIdentity, rund::kernel::ComputeApi,
             rund::kernel::ComputeScalar, rund::kernel::ComputeDomain,
             const DeviceVsmWindowFusion &,
             const DeviceVsmWindowRingPlan &) noexcept;
[[nodiscard]] bool
validate_geometry(const RangeExec &, const rund::kernel::WindowPlan &,
                  const DeviceVsmPageGeometry &, const DeviceVsmWindowFusion &,
                  const DeviceVsmWindowRingPlan &,
                  const DeviceVsmWindowMapSources &) noexcept;
[[nodiscard]] bool materialize_artifact(
    DeviceVsmWindowArtifact &, const RangeExec &,
    const rund::kernel::WindowPlan &, const DeviceVsmPageGeometry &,
    const DeviceVsmWindowFusion &, const DeviceVsmWindowRingPlan &,
    const rund::kernel::ArtifactKey &, rund::kernel::ComputeApi, std::string &&,
    std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool) noexcept;
[[nodiscard]] bool
emit_metal(const RangeExec &, const rund::kernel::ArtifactKey &,
           const DeviceVsmWindowFusion &, const DeviceVsmWindowRingPlan &,
           const DeviceVsmWindowMapSources &, std::string &) noexcept;
[[nodiscard]] bool
emit_vulkan(const RangeExec &, const rund::kernel::ArtifactKey &,
            const DeviceVsmWindowFusion &, const DeviceVsmWindowRingPlan &,
            const DeviceVsmWindowMapSources &, std::string &) noexcept;
[[nodiscard]] bool emit_metal_multipass(const RangeExec &,
                                        const rund::kernel::ArtifactKey &,
                                        const DeviceVsmWindowFusion &,
                                        const DeviceVsmWindowMapSources &,
                                        std::string &) noexcept;
[[nodiscard]] bool emit_vulkan_multipass(const RangeExec &,
                                         const rund::kernel::ArtifactKey &,
                                         const DeviceVsmWindowFusion &,
                                         const DeviceVsmWindowMapSources &,
                                         std::string &) noexcept;
void append_map_expression(std::string &, const DeviceVsmWindowMap &, MapSlot,
                           std::uint32_t, std::string_view, std::string_view);
void append_map_chain_expression(std::string &, const DeviceVsmWindowFusion &,
                                 MapSlot, std::string_view, std::string_view);
[[nodiscard]] bool
append_metal_map_functions(std::string &, const DeviceVsmWindowFusion &,
                           const DeviceVsmWindowMapSources &);
[[nodiscard]] bool
append_vulkan_map_functions(std::string &, const DeviceVsmWindowFusion &,
                            const DeviceVsmWindowMapSources &);

} // namespace rund::node::accel::detail::device_vsm_window_source
