#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

namespace vulkan_controlled_map_source_detail {

inline constexpr std::string_view Entry = "void main() {\n";
inline constexpr std::string_view DeclarationPrefix =
    "layout(set = 0, binding = ";
inline constexpr std::string_view DeclarationSuffix =
    ", std430) readonly buffer RundControlArgs { uint "
    "rund_control_args[]; };\n";
inline constexpr std::string_view Guard =
    "  if (gid >= rund_dispatch.tile_count) { return; }\n";
inline constexpr std::string_view ControlledGuard =
    "  if (gid >= rund_control_args[rund_dispatch.tile_count * 4u + 3u]) "
    "{ return; }\n";
inline constexpr std::string_view CanonicalVariant =
    "// artifact_variant=canonical";
inline constexpr std::string_view ControlledVariant =
    "// artifact_variant=controlled";

} // namespace vulkan_controlled_map_source_detail

[[nodiscard]] std::uint64_t
VulkanDecimalDigitCount(std::uint64_t value) noexcept;
[[nodiscard]] bool
VulkanControlledMapSourceUpperBytes(const rund::kernel::ComputePlan &plan,
                                    std::uint64_t specialized,
                                    std::uint64_t &upper) noexcept;

[[nodiscard]] bool VulkanMapControlSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] std::string VulkanMapControlSource();
[[nodiscard]] bool
VulkanGeneratedMapControlSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] std::string VulkanGeneratedMapControlSource();

[[nodiscard]] bool VulkanMapCheckSourceUpperBytes(
    std::uint64_t check_count, std::uint64_t offset_digit_bytes,
    std::uint64_t stride_digit_bytes, std::uint64_t limit_digit_bytes,
    std::uint64_t &upper) noexcept;
[[nodiscard]] bool
VulkanMapCheckSourceUpperBytes(const rund::kernel::LoweringArtifact &artifact,
                               std::uint64_t &upper) noexcept;

} // namespace rund::node::accel::detail
