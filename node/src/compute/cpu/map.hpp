#pragma once

#include "state.hpp"

#include <rund/compute/abi/model.hpp>

#include <memory>
#include <span>

namespace rund::compute::detail {

struct JobBufferView;

[[nodiscard]] Status
prepare_cpu_map_bindings(CpuProgram &program,
                         const std::shared_ptr<DeviceState> &device,
                         CpuMapRun &run, CpuMapRoute &route,
                         std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding>
                             reads,
                         std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding>
                             writes,
                         std::span<BufferState *const> inputs,
                         std::span<BufferState *const> outputs,
                         std::span<const JobBufferView> input_views,
                         std::span<const JobBufferView> output_views) noexcept;

[[nodiscard]] Status
begin_cpu_map(CpuProgram &program, CpuMapRun &run, CpuMapRoute &route,
              std::size_t logical_count, std::uint64_t &overflow_ordinal,
              const std::atomic_bool *cancel = nullptr) noexcept;

[[nodiscard]] kernel::ComputeTileCallbackResult
run_cpu_map_tile(const void *context, const kernel::ComputeTile &tile) noexcept;

} // namespace rund::compute::detail
