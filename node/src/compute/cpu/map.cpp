#include "map.hpp"
#include "view.hpp"

#include "../device/state.hpp"
#include "../job/state.hpp"
#include "../status.hpp"

#include <kernel/program/compute/plan.hpp>

#include "../../accel/cpu/simd/run/index.hpp"

#include <cstddef>
#include <limits>
#include <string_view>

namespace rund::compute::detail {

Status prepare_cpu_map_bindings(
    CpuProgram &program, const std::shared_ptr<DeviceState> &device,
    CpuMapRun &run, const std::span<BufferState *const> inputs,
    const std::span<BufferState *const> outputs,
    const std::span<const JobBufferView> input_views,
    const std::span<const JobBufferView> output_views) noexcept {
  const kernel::ComputeMap &map = program.map;
  kernel::u64 scalar_bytes = 0u;
  kernel::u64 summed_input_bytes = 0u;
  const std::size_t count = program.tile_plan.count();
  run.bindings_frozen = false;
  if (device == nullptr || cpu_device(*device) == nullptr || count == 0u ||
      inputs.size() > kernel::kMaxComputeBindingCount || outputs.empty() ||
      outputs.size() > MaxOutputs || inputs.size() != map.input_buffer_count ||
      outputs.size() != map.output_buffer_count ||
      input_views.size() != inputs.size() ||
      output_views.size() != outputs.size() ||
      program.input_bytes.size() != inputs.size() ||
      program.input_counts.size() != inputs.size() ||
      !kernel::ComputeScalarBytes(map.scalar, scalar_bytes) ||
      (outputs.size() == 1u
           ? !kernel::ComputeOutputBytesValid(map.scalar,
                                              map.output_bytes_per_tile)
           : map.output_bytes_per_tile != outputs.size() * scalar_bytes)) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  for (const std::uint64_t width : program.input_bytes) {
    if (summed_input_bytes >
        std::numeric_limits<std::uint64_t>::max() - width) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
    summed_input_bytes += width;
  }
  if (map.input_bytes_per_tile != summed_input_bytes) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  if (!run.tiles.prepared() || run.tiles.count() != count) {
    return Status::fail(Reason::TileBackendFailed);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    BufferState *const buffer = inputs[index];
    const JobBufferView view = input_views[index];
    if (view.count != program.input_counts[index] ||
        view.element_bytes != program.input_bytes[index]) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
    const std::optional<CpuView> bound = cpu_view(buffer, view);
    if (!bound || bound->data == nullptr) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
    run.reads[index] = node::accel::cpu_simd_detail::CpuSimdReadBinding{
        .data = reinterpret_cast<const unsigned char *>(bound->data),
        .stride = bound->footprint.stride,
    };
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    const kernel::u64 width =
        outputs.size() == 1u ? map.output_bytes_per_tile : scalar_bytes;
    BufferState *const buffer = outputs[index];
    const JobBufferView view = output_views[index];
    if (view.count != count || view.element_bytes != width) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
    const std::optional<CpuView> bound = cpu_view(buffer, view);
    if (!bound || bound->data == nullptr) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
    run.writes[index] = node::accel::cpu_simd_detail::CpuSimdWriteBinding{
        .data = reinterpret_cast<unsigned char *>(bound->data),
        .stride = bound->footprint.stride,
    };
  }
  run.bindings = node::accel::cpu_simd_detail::CpuSimdBindingView{
      .reads = run.reads.data(),
      .read_count = inputs.size(),
      .writes = run.writes.data(),
      .write_count = outputs.size(),
      .tile_count = count,
  };
  run.tile = CpuMapTileContext{.program = &program, .run = &run};
  return Status::success();
}

Status begin_cpu_map(CpuProgram &program, CpuMapRun &run,
                     const std::size_t logical_count,
                     std::uint64_t &overflow_ordinal,
                     const std::atomic_bool *const cancel) noexcept {
  const std::size_t count = program.tile_plan.count();
  if (logical_count > count ||
      logical_count > std::numeric_limits<kernel::u32>::max()) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  run.tile.cancel = cancel;
  run.bindings.tile_count = logical_count;
  const node::accel::cpu_simd_detail::IndexCheck indices =
      node::accel::cpu_simd_detail::ValidateIndices(
          run.bindings, program.read_routes, logical_count);
  if (!indices.ok()) {
    overflow_ordinal = std::min(overflow_ordinal, indices.ordinal);
    return Status::fail(std::string_view{indices.reason} ==
                                "compute_gather_index_out_of_range"
                            ? Reason::GatherIndexOutOfRange
                            : Reason::CpuBufferInvalid);
  }
  reset_simd(run);
  return Status::success();
}

kernel::ComputeTileCallbackResult
run_cpu_map_tile(const void *const raw,
                 const kernel::ComputeTile &tile) noexcept {
  const auto *const context = static_cast<const CpuMapTileContext *>(raw);
  if (context == nullptr || context->program == nullptr ||
      context->run == nullptr) {
    return {false, "compute_cpu_buffer_invalid"};
  }
  if (context->cancel != nullptr &&
      context->cancel->load(std::memory_order_acquire)) {
    return {false, "compute_cancelled"};
  }
  CpuProgram &cpu = *context->program;
  CpuMapRun &run = *context->run;
  const auto logical_count = static_cast<kernel::u32>(std::min<std::size_t>(
      run.bindings.tile_count, std::numeric_limits<kernel::u32>::max()));
  if (tile.begin >= logical_count) {
    return {};
  }
  const kernel::ComputeTile logical_tile{
      .index = tile.index,
      .worker_index = tile.worker_index,
      .begin = tile.begin,
      .end = std::min(tile.end, logical_count),
  };
  if (tile.worker_index >= run.scratch.size()) {
    return {false, "cpu_simd_scratch_invalid"};
  }
  const node::accel::CpuSimdRunResult result = cpu.dispatch.run(
      cpu.dispatch.prepared,
      node::accel::cpu_simd_detail::CpuSimdInvocation{
          .bindings = &run.bindings,
          .begin = logical_tile.begin,
          .count = logical_tile.size(),
      },
      node::accel::cpu_simd_detail::CpuSimdScratch{
          run.scratch[tile.worker_index].data(),
          run.scratch[tile.worker_index].size() * sizeof(std::max_align_t)});
  record_simd(run, tile.worker_index, result);
  return {result.ok, result.reason};
}

} // namespace rund::compute::detail
