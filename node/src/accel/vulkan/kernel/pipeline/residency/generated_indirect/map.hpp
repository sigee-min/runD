#pragma once

#include "../local.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

struct VulkanMapEncodeResources;

namespace vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct MapAccess final {
  bool valid{};
  bool direct{};
  bool selected{};
  bool graph{};
  bool recognized{};
  bool active{};
  bool descriptor_ready{};
  bool layout_ready{};
  bool resource_ready{};

  [[nodiscard]] constexpr bool usable() const noexcept {
    return recognized && valid &&
           (direct || (selected && active && descriptor_ready &&
                       layout_ready && resource_ready));
  }
};

[[nodiscard]] MapAccess
map_access(const VulkanResidencySelection &) noexcept;

void set_map_ready(VulkanResidencySelection &, bool) noexcept;

[[nodiscard]] rund::AccelCheck
prepare_gate(VulkanPipeline &pipeline,
             VulkanResidencySelection &selection) noexcept;

[[nodiscard]] bool
record_role(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
            const PersistentResidencySlidingRole &source, std::uint32_t stride,
            std::uint64_t plan_identity, std::uint64_t token,
            std::uint64_t run_generation, std::span<const std::uint32_t> locals,
            std::span<VkCommandBuffer> commands, std::size_t &command_count,
            std::uint64_t &dispatch_count, std::uint64_t &control_count,
            std::uint64_t &reset_count, std::uint64_t &reset_bytes,
            VulkanResidencyPersistentRun *expected_run, bool tail) noexcept;

void destroy_roles(VulkanResidencySelection &selection) noexcept;

[[nodiscard]] bool gate_result(VulkanResidencySlidingGate &gate,
                               std::uint64_t descriptor, bool &known_failure,
                               const char *&reason) noexcept;

#endif

} // namespace vulkan_generated_indirect_detail

} // namespace rund::node::accel::detail
