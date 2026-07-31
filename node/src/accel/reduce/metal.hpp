#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/reduce.hpp"

#include <kernel/program/compute/reduce/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;

[[nodiscard]] rund::AccelCheck ExecuteMetalReduce(
    const rund::AccelDevice &pick, const rund::kernel::ReduceDesc &desc,
    const rund::kernel::ReducePlan &plan, rund::kernel::ComputeDomain domain,
    const ReduceBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalReduce(
    const rund::AccelDevice &pick, const rund::kernel::ReduceDesc &desc,
    const rund::kernel::ReducePlan &plan, rund::kernel::ComputeDomain domain,
    const ReduceBinds &bindings, std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck
EncodeMetalReduce(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
                  void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalReduce(MetalAdapter &adapter,
                  const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
