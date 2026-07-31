#pragma once

#include "step.hpp"
#include "../cpu/state.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <rund/compute/abi/model.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] Result<compute_dsl::ComputeOp>
build_map_operation_multi(std::size_t count, std::span<const Type> outputs,
                          std::span<const Type> inputs,
                          std::span<const ExprRef> expressions,
                          std::span<const MapRead> reads = {});

[[nodiscard]] Result<std::unique_ptr<CpuProgram>>
prepare_cpu_map(const std::shared_ptr<DeviceState> &device, std::size_t count,
                std::span<const Type> outputs, std::span<const Type> inputs,
                compute_dsl::ComputeOp operation);

} // namespace rund::compute::detail
