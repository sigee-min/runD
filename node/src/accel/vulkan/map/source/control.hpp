#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanMapTemplateResources;

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
VulkanMapCheckHash(const VulkanMapTemplateResources &) noexcept;

[[nodiscard]] rund::kernel::LoweringArtifact
VulkanMapCheckArtifact(const VulkanMapTemplateResources &);

[[nodiscard]] rund::kernel::LoweringArtifact
VulkanMapControlArtifact(const rund::kernel::ComputePlan &);

[[nodiscard]] rund::kernel::LoweringArtifact
VulkanGeneratedMapControlArtifact(const rund::kernel::ComputePlan &);

[[nodiscard]] rund::kernel::LoweringArtifact
VulkanControlledMapArtifact(rund::kernel::LoweringArtifact,
                            const rund::kernel::ComputePlan &);

#endif

} // namespace rund::node::accel::detail
