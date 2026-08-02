#include "../../../accel/compact.hpp"
#include "../../../accel/cpu/buffer.hpp"
#include "../../../accel/cpu/collective.hpp"
#include "../../../accel/cpu/scatter/linear.hpp"
#include "../../../accel/cpu/sort/radix.hpp"
#include "../../../accel/gather.hpp"
#include "../../../accel/histogram.hpp"
#include "../../../accel/partition.hpp"
#include "../../../accel/scatter.hpp"
#include "../../../accel/sort.hpp"
#include "../../backend.hpp"
#include "../../host.hpp"
#include "../../program/state.hpp"
#include "../bounded.hpp"
#include "../graph.hpp"
#include "../scratch.hpp"
#include "../view.hpp"
#include "primitive/algebra.hpp"
#include "state.hpp"

#include <accel/check.hpp>
#include <kernel/program/compute/compact/reference.hpp>
#include <kernel/program/compute/gather/reference.hpp>
#include <kernel/program/compute/histogram/reference.hpp>
#include <kernel/program/compute/partition/reference.hpp>
#include <kernel/program/compute/scatter/reduce/reference.hpp>
#include <kernel/program/compute/segmented/reduce/reference.hpp>
#include <kernel/program/compute/segmented/scan/reference.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <utility>

namespace rund::compute::detail {

namespace {

[[nodiscard]] RawCpuBuffer raw_cpu_buffer(BufferState *const buffer,
                                          const JobBufferView view) noexcept {
  const std::optional<CpuView> bound = cpu_view(buffer, view);
  return !bound || bound->data == nullptr || !bound->footprint.dense()
             ? RawCpuBuffer{}
             : RawCpuBuffer{.data = bound->data,
                            .bytes = bound->footprint.bytes};
}

[[nodiscard]] Status run_cpu_primitive(JobState &job, CpuGraphProgram &cpu,
                                       CpuGraphRun &run,
                                       const std::size_t step_index,
                                       const CpuRuntimePrimitive &primitive) {
  const std::shared_ptr<ProgramState> &program = job.program;
  const auto &inputs = job.inputs;
  const auto &outputs = job.outputs;
  constexpr std::size_t kMaxPorts = 8u;
  std::array<RawCpuBuffer, kMaxPorts> ports{};
  const std::size_t begin = cpu.bind_begin[step_index];
  const std::size_t count = cpu.bind_count[step_index];
  if (count == 0u || count > ports.size() ||
      begin > program->graph_bindings.size() ||
      count > program->graph_bindings.size() - begin) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const GraphRunBinding &binding = program->graph_bindings[begin + index];
    BufferState *const buffer =
        graph_binding_buffer(*program, binding, inputs, outputs, run.buffers);
    if (buffer == nullptr) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    ports[index] =
        raw_cpu_buffer(buffer, job_binding_view(job, binding, *buffer));
    if (!ports[index]) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
  }
  const auto port = [&](const std::size_t index) -> const RawCpuBuffer & {
    return ports[index];
  };
  const auto bounded_count = [&](const std::uint32_t value,
                                 const std::size_t capacity,
                                 kernel::u32 &count) noexcept {
    if (value == 0u || value > cpu.runtime->values.size()) {
      return Status::fail(Reason::BoundedCountInvalid);
    }
    BufferState *const buffer =
        graph_value_buffer_id(*program, value, inputs, outputs, run.buffers);
    const JobBufferView view = buffer == nullptr
                                   ? JobBufferView{}
                                   : job_value_view(job, value, *buffer);
    return read_bounded_count(
        buffer, view, cpu.runtime->values[value - 1u].type, capacity, count);
  };
  rund::AccelCheck check{false, "compute_primitive_unsupported"};
  switch (primitive.kind) {
  case Primitive::SegmentedScan: {
    const auto &plan = std::get<kernel::SegmentedScanPlan>(primitive.plan);
    const CpuRuntimeGraph &graph = *program->cpu_graph->runtime;
    const Type value_type = graph.values[primitive.inputs.front() - 1u].type;
    const auto *heads = reinterpret_cast<const kernel::u32 *>(port(1u).data);
    kernel::SegmentedScanResult result{};
    if (value_type == Type::I32 || value_type == Type::FixedLane32) {
      result = kernel::ReferenceSignedSegmentedScan(
          reinterpret_cast<const std::int32_t *>(port(0u).data), heads,
          reinterpret_cast<std::int32_t *>(port(2u).data), plan.element_count,
          plan.op == kernel::SegmentedScanOp::InclusiveSum);
    } else if (value_type == Type::I64 || value_type == Type::FixedLane64) {
      result = kernel::ReferenceSignedSegmentedScan(
          reinterpret_cast<const std::int64_t *>(port(0u).data), heads,
          reinterpret_cast<std::int64_t *>(port(2u).data), plan.element_count,
          plan.op == kernel::SegmentedScanOp::InclusiveSum);
    } else if (plan.element == kernel::SegmentedScanElement::U32 &&
               plan.op == kernel::SegmentedScanOp::ExclusiveSum) {
      result = kernel::ReferenceExclusiveSegmentedScanU32(
          reinterpret_cast<const kernel::u32 *>(port(0u).data), heads,
          reinterpret_cast<kernel::u32 *>(port(2u).data), plan.element_count);
    } else if (plan.element == kernel::SegmentedScanElement::U32) {
      result = kernel::ReferenceInclusiveSegmentedScanU32(
          reinterpret_cast<const kernel::u32 *>(port(0u).data), heads,
          reinterpret_cast<kernel::u32 *>(port(2u).data), plan.element_count);
    } else if (plan.op == kernel::SegmentedScanOp::ExclusiveSum) {
      result = kernel::ReferenceExclusiveSegmentedScanU64(
          reinterpret_cast<const kernel::u64 *>(port(0u).data), heads,
          reinterpret_cast<kernel::u64 *>(port(2u).data), plan.element_count);
    } else {
      result = kernel::ReferenceInclusiveSegmentedScanU64(
          reinterpret_cast<const kernel::u64 *>(port(0u).data), heads,
          reinterpret_cast<kernel::u64 *>(port(2u).data), plan.element_count);
    }
    check = rund::AccelCheck{result.ok, result.reason};
    break;
  }
  case Primitive::SegmentedReduce: {
    const auto &plan = std::get<kernel::SegmentedReducePlan>(primitive.plan);
    const auto op = plan.op;
    const CpuRuntimeGraph &graph = *program->cpu_graph->runtime;
    const Type value_type = graph.values[primitive.inputs.front() - 1u].type;
    const auto *heads = reinterpret_cast<const kernel::u32 *>(port(1u).data);
    kernel::SegmentedReduceResult result{};
    if (value_type == Type::I32 || value_type == Type::FixedLane32) {
      result = kernel::ReferenceSignedSegmentedReduce(
          reinterpret_cast<const std::int32_t *>(port(0u).data), heads,
          reinterpret_cast<std::int32_t *>(port(2u).data), plan.element_count,
          op);
    } else if (value_type == Type::I64 || value_type == Type::FixedLane64) {
      result = kernel::ReferenceSignedSegmentedReduce(
          reinterpret_cast<const std::int64_t *>(port(0u).data), heads,
          reinterpret_cast<std::int64_t *>(port(2u).data), plan.element_count,
          op);
    } else if (plan.element == kernel::ReduceElement::U32) {
      const auto *values = reinterpret_cast<const kernel::u32 *>(port(0u).data);
      auto *output = reinterpret_cast<kernel::u32 *>(port(2u).data);
      switch (op) {
      case kernel::ReduceOp::Sum:
        result = kernel::ReferenceSegmentedReduceSumU32(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::CountNonzero:
        result = kernel::ReferenceSegmentedReduceCountNonzeroU32(
            values, heads, output, plan.element_count);
        break;
      case kernel::ReduceOp::Min:
        result = kernel::ReferenceSegmentedReduceMinU32(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::Max:
        result = kernel::ReferenceSegmentedReduceMaxU32(values, heads, output,
                                                        plan.element_count);
        break;
      }
    } else {
      const auto *values = reinterpret_cast<const kernel::u64 *>(port(0u).data);
      auto *output = reinterpret_cast<kernel::u64 *>(port(2u).data);
      switch (op) {
      case kernel::ReduceOp::Sum:
        result = kernel::ReferenceSegmentedReduceSumU64(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::CountNonzero:
        result = kernel::ReferenceSegmentedReduceCountNonzeroU64(
            values, heads, output, plan.element_count);
        break;
      case kernel::ReduceOp::Min:
        result = kernel::ReferenceSegmentedReduceMinU64(values, heads, output,
                                                        plan.element_count);
        break;
      case kernel::ReduceOp::Max:
        result = kernel::ReferenceSegmentedReduceMaxU64(values, heads, output,
                                                        plan.element_count);
        break;
      }
    }
    check = rund::AccelCheck{result.ok, result.reason};
    break;
  }
  case Primitive::Sort:
  case Primitive::Argsort: {
    const Type type = cpu.runtime->values[primitive.inputs.front() - 1u].type;
    const bool signed_order = type == Type::I32 || type == Type::I64 ||
                              type == Type::FixedLane32 ||
                              type == Type::FixedLane64;
    const auto &plan = std::get<kernel::SortPlan>(primitive.plan);
    kernel::u32 active_count = static_cast<kernel::u32>(plan.element_count);
    const bool bounded = primitive.inputs.size() == 2u;
    if (bounded) {
      const Status count_status =
          bounded_count(primitive.inputs[1u], plan.element_count, active_count);
      if (!count_status) {
        return Status::fail(count_status.reason());
      }
    }
    const std::size_t key_output = bounded ? 2u : 1u;
    const std::size_t value_output = key_output + 1u;
    if (plan.key == kernel::SortKey::U32) {
      auto *const scratch =
          cpu_primitive_scratch<CpuSortPrimitiveScratch<kernel::u32>>(
              cpu_step_scratch(run, step_index));
      check =
          scratch == nullptr
              ? rund::AccelCheck{false, "compute_sort_invalid"}
              : node::accel::detail::ExecuteCpuRadixSortPrepared(
                    reinterpret_cast<const kernel::u32 *>(port(0u).data),
                    nullptr,
                    reinterpret_cast<kernel::u32 *>(port(key_output).data),
                    reinterpret_cast<kernel::u32 *>(port(value_output).data),
                    active_count, plan.radix_pass_count, true, signed_order,
                    scratch->keys.data(), scratch->values.data(),
                    scratch->counts, scratch->offsets);
    } else {
      auto *const scratch =
          cpu_primitive_scratch<CpuSortPrimitiveScratch<kernel::u64>>(
              cpu_step_scratch(run, step_index));
      check =
          scratch == nullptr
              ? rund::AccelCheck{false, "compute_sort_invalid"}
              : node::accel::detail::ExecuteCpuRadixSortPrepared(
                    reinterpret_cast<const kernel::u64 *>(port(0u).data),
                    nullptr,
                    reinterpret_cast<kernel::u64 *>(port(key_output).data),
                    reinterpret_cast<kernel::u32 *>(port(value_output).data),
                    active_count, plan.radix_pass_count, true, signed_order,
                    scratch->keys.data(), scratch->values.data(),
                    scratch->counts, scratch->offsets);
    }
    break;
  }
  case Primitive::Compact: {
    const auto &plan = std::get<kernel::CompactPlan>(primitive.plan);
    kernel::u64 output_count = 0u;
    const auto result = kernel::ReferenceCompactIdsU32(
        reinterpret_cast<const kernel::u32 *>(port(0u).data),
        plan.element_count, plan.output_capacity,
        reinterpret_cast<kernel::u32 *>(port(1u).data), &output_count);
    check = rund::AccelCheck{result.ok, result.reason};
    break;
  }
  case Primitive::Gather: {
    const auto &plan = std::get<kernel::GatherPlan>(primitive.plan);
    kernel::u32 logical_count = static_cast<kernel::u32>(plan.element_count);
    const bool bounded = primitive.inputs.size() == 3u;
    if (bounded) {
      const Status count_status = bounded_count(
          primitive.inputs[2u], plan.element_count, logical_count);
      if (!count_status) {
        return count_status;
      }
    }
    const std::size_t output = bounded ? 3u : 2u;
    const auto result =
        plan.element == kernel::GatherElement::U32
            ? kernel::ReferenceGatherU32(
                  reinterpret_cast<const kernel::u32 *>(port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(port(1u).data),
                  reinterpret_cast<kernel::u32 *>(port(output).data),
                  logical_count, plan.source_count)
            : kernel::ReferenceGatherU64(
                  reinterpret_cast<const kernel::u64 *>(port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(port(1u).data),
                  reinterpret_cast<kernel::u64 *>(port(output).data),
                  logical_count, plan.source_count);
    check = rund::AccelCheck{result.ok, result.reason};
    if (!result.ok) {
      run.overflow_ordinal =
          std::min(run.overflow_ordinal, result.first_invalid_index);
    }
    break;
  }
  case Primitive::Histogram: {
    const auto &plan = std::get<kernel::HistogramPlan>(primitive.plan);
    const auto result = kernel::ReferenceHistogramU32(
        reinterpret_cast<const kernel::u32 *>(port(0u).data),
        reinterpret_cast<kernel::u32 *>(port(1u).data), plan.element_count,
        plan.bin_count);
    check = rund::AccelCheck{result.ok, result.reason};
    break;
  }
  case Primitive::Partition: {
    const auto &plan = std::get<kernel::PartitionPlan>(primitive.plan);
    kernel::u64 false_count = 0u;
    kernel::u64 true_count = 0u;
    const bool wide_flags = plan.flag_bytes == sizeof(kernel::u64);
    const bool wide_values = plan.value_bytes == sizeof(kernel::u64);
    const auto result =
        wide_flags && wide_values
            ? kernel::ReferenceStablePartitionFlagsU64ValuesU64(
                  reinterpret_cast<const kernel::u64 *>(port(0u).data),
                  reinterpret_cast<const kernel::u64 *>(port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u64 *>(port(2u).data), &false_count,
                  &true_count)
        : wide_flags ? kernel::ReferenceStablePartitionFlagsU64ValuesU32(
                           reinterpret_cast<const kernel::u64 *>(port(0u).data),
                           reinterpret_cast<const kernel::u32 *>(port(1u).data),
                           plan.element_count,
                           reinterpret_cast<kernel::u32 *>(port(2u).data),
                           &false_count, &true_count)
        : wide_values
            ? kernel::ReferenceStablePartitionU64(
                  reinterpret_cast<const kernel::u32 *>(port(0u).data),
                  reinterpret_cast<const kernel::u64 *>(port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u64 *>(port(2u).data), &false_count,
                  &true_count)
            : kernel::ReferenceStablePartitionU32(
                  reinterpret_cast<const kernel::u32 *>(port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(port(1u).data),
                  plan.element_count,
                  reinterpret_cast<kernel::u32 *>(port(2u).data), &false_count,
                  &true_count);
    check = rund::AccelCheck{result.ok, result.reason};
    break;
  }
  case Primitive::Scatter: {
    const auto &plan = std::get<kernel::ScatterPlan>(primitive.plan);
    CpuScatterPrimitiveScratch *const scratch =
        cpu_primitive_scratch<CpuScatterPrimitiveScratch>(
            cpu_step_scratch(run, step_index));
    if (scratch == nullptr) {
      check = rund::AccelCheck{false, "compute_scatter_invalid"};
      break;
    }
    const auto result =
        plan.element == kernel::ScatterElement::U32
            ? node::accel::detail::ExecuteLinearScatter(
                  *scratch,
                  reinterpret_cast<const kernel::u32 *>(port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(port(1u).data),
                  reinterpret_cast<kernel::u32 *>(port(2u).data),
                  plan.element_count, plan.output_count,
                  scratch->keys.size())
            : node::accel::detail::ExecuteLinearScatter(
                  *scratch,
                  reinterpret_cast<const kernel::u64 *>(port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(port(1u).data),
                  reinterpret_cast<kernel::u64 *>(port(2u).data),
                  plan.element_count, plan.output_count,
                  scratch->keys.size());
    check = rund::AccelCheck{result.ok, result.reason};
    break;
  }
  case Primitive::ScatterReduce: {
    const auto &plan = std::get<kernel::ScatterReducePlan>(primitive.plan);
    auto *const scratch =
        cpu_primitive_scratch<CpuScatterReducePrimitiveScratch>(
            cpu_step_scratch(run, step_index));
    if (scratch == nullptr) {
      check = rund::AccelCheck{false, "compute_scatter_reduce_buffer_invalid"};
      break;
    }
    kernel::u32 logical_count = static_cast<kernel::u32>(plan.element_count);
    const bool bounded = primitive.inputs.size() == 3u;
    if (bounded) {
      const Status count_status = bounded_count(
          primitive.inputs[2u], plan.element_count, logical_count);
      if (!count_status) {
        run.overflow_ordinal =
            std::min(run.overflow_ordinal,
                     static_cast<std::uint64_t>(plan.element_count));
        return count_status;
      }
    }
    const RawCpuBuffer &values = port(0u);
    const RawCpuBuffer &indices = port(1u);
    const RawCpuBuffer &output = port(bounded ? 3u : 2u);
    kernel::ScatterReduceResult result{};
    switch (plan.domain) {
    case kernel::ComputeDomain::I32:
      result = kernel::ReferenceScatterReduceI32(
          reinterpret_cast<const kernel::i32 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::i32 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::U32:
      result = kernel::ReferenceScatterReduceU32(
          reinterpret_cast<const kernel::u32 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::u32 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::I64:
      result = kernel::ReferenceScatterReduceI64(
          reinterpret_cast<const kernel::i64 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::i64 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::U64:
      result = kernel::ReferenceScatterReduceU64(
          reinterpret_cast<const kernel::u64 *>(values.data),
          reinterpret_cast<const kernel::u32 *>(indices.data),
          reinterpret_cast<kernel::u64 *>(output.data), logical_count, plan,
          scratch->sorted_indices.data(), scratch->sorted_indices.size());
      break;
    case kernel::ComputeDomain::Fixed:
      result = plan.element_bytes == sizeof(kernel::i64)
                   ? kernel::ReferenceScatterReduceFixedI64(
                         reinterpret_cast<const kernel::i64 *>(values.data),
                         reinterpret_cast<const kernel::u32 *>(indices.data),
                         reinterpret_cast<kernel::i64 *>(output.data),
                         logical_count, plan, scratch->sorted_indices.data(),
                         scratch->sorted_indices.size())
                   : kernel::ReferenceScatterReduceFixedI32(
                         reinterpret_cast<const kernel::i32 *>(values.data),
                         reinterpret_cast<const kernel::u32 *>(indices.data),
                         reinterpret_cast<kernel::i32 *>(output.data),
                         logical_count, plan, scratch->sorted_indices.data(),
                         scratch->sorted_indices.size());
      break;
    }
    check = rund::AccelCheck{result.ok, result.reason};
    if (result.ok) {
      run.conflict_count = ::rund::detail::counter::SaturatingAdd(
          run.conflict_count, result.conflict_count);
    } else {
      run.overflow_ordinal =
          std::min(run.overflow_ordinal, result.first_rejected_ordinal);
    }
    break;
  }
  case Primitive::Stencil: {
    PrimitiveContext context{
        job,       cpu,
        run,       step_index,
        primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
    return run_stencil(context);
  }
  case Primitive::Transform: {
    PrimitiveContext context{
        job,       cpu,
        run,       step_index,
        primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
    return run_transform(context);
  }
  case Primitive::Matrix: {
    PrimitiveContext context{
        job,       cpu,
        run,       step_index,
        primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
    return run_matrix(context);
  }
  case Primitive::Factor: {
    PrimitiveContext context{
        job,       cpu,
        run,       step_index,
        primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
    return run_factor(context);
  }
  case Primitive::Solve: {
    PrimitiveContext context{
        job,       cpu,
        run,       step_index,
        primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
    return run_solve(context);
  }
  case Primitive::Spectrum: {
    PrimitiveContext context{
        job,       cpu,
        run,       step_index,
        primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
    return run_spectrum(context);
  }
  case Primitive::Reduce:
    return Status::fail(Reason::ReduceRouteInvalid);
  }
  return check.ok ? Status::success()
                  : Status::fail(project_reason(
                        check.reason, Reason::PrimitiveBackendFailed));
}

} // namespace

Status execute_cpu_primitive(JobState &job, const std::size_t step) {
  if (job.program == nullptr || job.program->cpu_graph == nullptr ||
      job.cpu == nullptr || job.cpu->graph == nullptr ||
      job.cpu->graph->storage == nullptr || job.inputs.empty() ||
      job.outputs.empty()) {
    return Status::fail(Reason::RunInvalid);
  }
  CpuGraphProgram &program = *job.program->cpu_graph;
  const CpuRuntimeGraph &graph = *program.runtime;
  if (step >= graph.steps.size()) {
    return Status::fail(Reason::GraphStepInvalid);
  }
  const auto *const primitive =
      std::get_if<CpuRuntimePrimitive>(&graph.steps[step]);
  if (primitive == nullptr || primitive->kind == Primitive::Reduce) {
    return Status::fail(Reason::PrimitiveRouteInvalid);
  }
  return run_cpu_primitive(job, program, *job.cpu->graph, step, *primitive);
}

} // namespace rund::compute::detail
