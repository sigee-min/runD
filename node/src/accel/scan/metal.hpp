#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "metal/buffers.hpp"
#include <kernel/program/compute/graph/schema.hpp>

namespace rund::node::accel::detail {

struct MetalRuntimeBuffer;

[[nodiscard]] rund::AccelCheck
ExecuteMetalScan(const rund::AccelDevice &pick,
                 const rund::kernel::ScanDesc &desc,
                 const rund::kernel::ScanPlan &plan,
                 rund::kernel::ComputeDomain domain, const ScanBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalScan(
    const rund::AccelDevice &pick, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    const ScanBinds &bindings, std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck
EncodeMetalScan(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
                void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalScan(MetalAdapter &adapter, const std::shared_ptr<void> &resources);

[[nodiscard]] bool MetalScanStatusOk(const MetalRuntimeBuffer &status);
[[nodiscard]] rund::kernel::u32
MetalScanStatusFlags(const MetalRuntimeBuffer &status);

} // namespace rund::node::accel::detail
