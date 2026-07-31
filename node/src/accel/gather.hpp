#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "kernel/bindings/gather.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/gather/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct VulkanAdapter;

[[nodiscard]] rund::AccelCheck ExecuteMetalGather(const rund::AccelDevice &pick,
                                       const rund::kernel::GatherDesc &desc,
                                       const rund::kernel::GatherPlan &plan,
                                       const GatherBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalGather(const rund::AccelDevice &pick,
                                       const rund::kernel::GatherDesc &desc,
                                       const rund::kernel::GatherPlan &plan,
                                       const GatherBinds &bindings,
                                       std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck EncodeMetalGather(MetalAdapter &adapter,
                                      const std::shared_ptr<void> &resources,
                                      void *command_encoder);
[[nodiscard]] rund::AccelCheck FinishMetalGather(MetalAdapter &adapter,
                                      const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck ExecuteVulkanGather(const rund::AccelDevice &pick,
                                        const rund::kernel::GatherDesc &desc,
                                        const rund::kernel::GatherPlan &plan,
                                        const GatherBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareVulkanGather(const rund::AccelDevice &pick,
                                        const rund::kernel::GatherDesc &desc,
                                        const rund::kernel::GatherPlan &plan,
                                        const GatherBinds &bindings,
                                        std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck EncodeVulkanGather(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources,
                                       void *command_buffer);
[[nodiscard]] rund::AccelCheck FinishVulkanGather(VulkanAdapter &adapter,
                                       const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
