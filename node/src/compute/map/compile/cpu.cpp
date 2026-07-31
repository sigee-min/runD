#include "../build.hpp"

#include "../../device/state.hpp"
#include "../../status.hpp"
#include "../../type.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace rund::compute::detail {

Result<std::unique_ptr<CpuProgram>>
prepare_cpu_map(const std::shared_ptr<DeviceState> &device,
                const std::size_t count, const std::span<const Type> outputs,
                const std::span<const Type> inputs,
                compute_dsl::ComputeOp operation) {
  CpuDeviceState *const host =
      device == nullptr ? nullptr : cpu_device(*device);
  if (host == nullptr || !host->workers ||
      count > std::numeric_limits<kernel::u32>::max()) {
    return Result<std::unique_ptr<CpuProgram>>::fail(Reason::ShapeCapacity);
  }
  if (!operation.ok()) {
    return Result<std::unique_ptr<CpuProgram>>::fail(
        project_reason(operation.reason(), Reason::IrInvalid));
  }
  try {
    auto prepared = std::make_unique<CpuProgram>();
    CpuProgram &cpu = *prepared;
    cpu.map = operation.map();
    cpu.map.api = kernel::ComputeApi::Cpu;
    const compute_dsl::detail::ComputeOpMetadataView metadata =
        operation.metadata_view();
    if (metadata.input_count != inputs.size() ||
        metadata.output_count != outputs.size() ||
        metadata.param_bytes != cpu.map.param_bytes ||
        (metadata.input_count != 0u &&
         metadata.input_element_bytes == nullptr) ||
        metadata.output_element_bytes == nullptr ||
        (metadata.param_bytes != 0u && metadata.param_data == nullptr)) {
      return Result<std::unique_ptr<CpuProgram>>::fail(
          Reason::IrBindingInvalid);
    }
    cpu.input_bytes.assign(metadata.input_element_bytes,
                           metadata.input_element_bytes +
                               metadata.input_count);
    const kernel::ExecutionMetadata execution =
        kernel::BuildExecutionMetadata(operation.ir(), kernel::ComputeApi::Cpu);
    if (!execution) {
      return Result<std::unique_ptr<CpuProgram>>::fail(
          Reason::IrBindingInvalid);
    }
    cpu.read_routes = execution.read_routes;
    cpu.input_counts.resize(inputs.size());
    std::array<std::uint64_t, kernel::kMaxComputeBindingCount> input_probes{};
    std::array<std::uint64_t, MaxOutputs> output_probes{};
    std::array<kernel::BufferSpan, kernel::kMaxComputeBindingCount>
        input_spans{};
    std::array<kernel::OutputSpan, MaxOutputs> output_spans{};
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
      const kernel::u64 width = metadata.input_element_bytes[index];
      if (width != type_bytes(inputs[index])) {
        return Result<std::unique_ptr<CpuProgram>>::fail(
            Reason::IrBindingInvalid);
      }
      const kernel::u64 required = kernel::RequiredInputCount(
          execution, static_cast<kernel::u64>(index), count);
      if (required == 0u) {
        return Result<std::unique_ptr<CpuProgram>>::fail(
            Reason::IrBindingInvalid);
      }
      cpu.input_counts[index] = required;
      input_spans[index] = kernel::BufferSpan{.data = &input_probes[index],
                                              .element_bytes = width,
                                              .stride_bytes = width,
                                              .count = required};
    }
    for (const kernel::ReadRoute route : cpu.read_routes) {
      if (route.source >= inputs.size() || route.index >= inputs.size()) {
        return Result<std::unique_ptr<CpuProgram>>::fail(
            Reason::IrBindingInvalid);
      }
    }
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
      const kernel::u64 width = metadata.output_element_bytes[index];
      if (width != type_bytes(outputs[index])) {
        return Result<std::unique_ptr<CpuProgram>>::fail(
            Reason::IrBindingInvalid);
      }
      output_spans[index] = kernel::OutputSpan{.data = &output_probes[index],
                                               .element_bytes = width,
                                               .stride_bytes = width,
                                               .count = count};
    }
    const kernel::ComputeMap &map = cpu.map;
    const kernel::BindingSet bindings{
        .phase_id = 1u,
        .tile_count = count,
        .lane_count = count,
        .op_hash_hi = map.op_hash_hi,
        .op_hash_lo = map.op_hash_lo,
        .api = kernel::ComputeApi::Cpu,
        .scalar = map.scalar,
        .domain = map.domain,
        .input_bytes_per_tile = map.input_bytes_per_tile,
        .output_bytes_per_tile = map.output_bytes_per_tile,
        .param_bytes = map.param_bytes,
        .metadata_bytes_per_tile = map.metadata_bytes_per_tile,
        .input_buffers = input_spans.data(),
        .input_buffer_count = inputs.size(),
        .output_buffers = output_spans.data(),
        .output_buffer_count = outputs.size(),
        .input_element_bytes = metadata.input_element_bytes,
        .input_element_byte_count = metadata.input_count,
        .output_element_bytes = metadata.output_element_bytes,
        .output_element_byte_count = metadata.output_count,
        .param_data = metadata.param_data,
        .param_data_bytes = metadata.param_bytes,
        .ok = true,
        .reason = "ok"};
    cpu.dispatch = node::accel::cpu_simd_detail::PrepareCpuSimdDispatch(
        operation.ir(), host->caps, bindings);
    if (!cpu.dispatch.prepared.ok || cpu.dispatch.run == nullptr ||
        cpu.dispatch.scratch_bytes == nullptr) {
      return Result<std::unique_ptr<CpuProgram>>::fail(project_reason(
          cpu.dispatch.prepared.reason, Reason::LoweringInvalid));
    }
    cpu.tile_plan = kernel::ComputeTileExecutor{
        host->workers.backend, host->workers.requested_worker_width};
    const kernel::ComputeTilePrepareResult tiles =
        cpu.tile_plan.prepare(static_cast<kernel::u32>(count));
    if (!tiles.ok) {
      return Result<std::unique_ptr<CpuProgram>>::fail(
          project_reason(tiles.reason, Reason::TileBackendFailed));
    }
    const std::size_t scratch_bytes =
        cpu.dispatch.scratch_bytes(cpu.dispatch.prepared);
    cpu.scratch_words = (scratch_bytes + sizeof(std::max_align_t) - 1u) /
                        sizeof(std::max_align_t);
    cpu.workers = tiles.worker_count;
    cpu.tile_size = tiles.tile_units;
    return Result<std::unique_ptr<CpuProgram>>::success(std::move(prepared));
  } catch (const std::bad_alloc &) {
    return Result<std::unique_ptr<CpuProgram>>::fail(Reason::ProgramCapacity);
  }
}

} // namespace rund::compute::detail
