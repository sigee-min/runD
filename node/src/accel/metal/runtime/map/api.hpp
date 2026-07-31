#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct BoundControl;

[[nodiscard]] rund::AccelCheck PrepareMetalMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<void> &resources, std::uint32_t iterations = 1u);
[[nodiscard]] rund::AccelCheck
EncodeMetalMap(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
               void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalMap(MetalAdapter &adapter, const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
