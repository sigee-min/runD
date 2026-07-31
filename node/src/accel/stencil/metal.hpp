#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/stencil.hpp"
#include <kernel/program/compute/stencil/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;

[[nodiscard]] rund::AccelCheck ExecuteMetalStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan, rund::kernel::ComputeDomain domain,
    const StencilBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan, rund::kernel::ComputeDomain domain,
    const StencilBinds &bindings, std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck
EncodeMetalStencil(MetalAdapter &adapter,
                   const std::shared_ptr<void> &resources,
                   void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalStencil(MetalAdapter &adapter,
                   const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
