#include "local.hpp"

#include "../../backend.hpp"
#include "../../cpu/graph.hpp"

#include <kernel/program/compute/compact/plan.hpp>
#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/gather/plan.hpp>
#include <kernel/program/compute/histogram/plan.hpp>
#include <kernel/program/compute/matrix/plan.hpp>
#include <kernel/program/compute/partition/plan.hpp>
#include <kernel/program/compute/reduce/plan.hpp>
#include <kernel/program/compute/scatter/plan.hpp>
#include <kernel/program/compute/scatter/reduce/plan.hpp>
#include <kernel/program/compute/segmented/reduce/plan.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>
#include <kernel/program/compute/solve/plan.hpp>
#include <kernel/program/compute/sort/plan.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>
#include <kernel/program/compute/stencil/plan.hpp>
#include <kernel/program/compute/transform/plan.hpp>

#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>

namespace rund::compute::detail::graph_compile {
namespace {

[[nodiscard]] Status plan(CpuRuntimePrimitive &runtime,
                          const GraphPrimitive &primitive) {
  switch (primitive.primitive) {
  case Primitive::SegmentedScan:
    runtime.plan = kernel::PlanSegmentedScan(primitive.node.segmented_scan);
    break;
  case Primitive::SegmentedReduce:
    runtime.plan = kernel::PlanSegmentedReduce(primitive.node.segmented_reduce);
    break;
  case Primitive::Sort:
  case Primitive::Argsort:
    runtime.plan = kernel::PlanSort(primitive.node.sort);
    break;
  case Primitive::Compact:
    runtime.plan = kernel::PlanCompact(primitive.node.compact);
    break;
  case Primitive::Gather:
    runtime.plan = kernel::PlanGather(primitive.node.gather);
    break;
  case Primitive::Histogram:
    runtime.plan = kernel::PlanHistogram(primitive.node.histogram);
    break;
  case Primitive::Partition:
    runtime.plan = kernel::PlanPartition(primitive.node.partition);
    break;
  case Primitive::Reduce:
    runtime.plan = kernel::PlanReduce(primitive.node.reduce);
    break;
  case Primitive::Scatter:
    runtime.plan = kernel::PlanScatter(primitive.node.scatter);
    break;
  case Primitive::ScatterReduce:
    runtime.plan = kernel::PlanScatterReduce(primitive.node.scatter_reduce);
    break;
  case Primitive::Stencil:
    runtime.plan = kernel::PlanStencil(primitive.node.stencil);
    break;
  case Primitive::Transform:
    runtime.plan = kernel::PlanTransform(primitive.node.transform);
    break;
  case Primitive::Matrix:
    runtime.plan = kernel::PlanMatrix(primitive.node.matrix);
    break;
  case Primitive::Factor:
    runtime.plan = kernel::PlanFactor(primitive.node.factor);
    break;
  case Primitive::Solve:
    runtime.plan = kernel::PlanSolve(primitive.node.solve);
    break;
  case Primitive::Spectrum:
    runtime.plan = kernel::PlanSpectrum(primitive.node.spectrum);
    break;
  }
  const char *const reason = std::visit(
      [](const auto &prepared) {
        return prepared.ok ? static_cast<const char *>(nullptr)
                           : prepared.reason;
      },
      runtime.plan);
  return reason == nullptr
             ? Status::success()
             : Status::fail(project_reason(reason, Reason::LoweringInvalid));
}

} // namespace

Status primitive(Lowering &lowering, const std::size_t index,
                 const GraphPrimitive &step) {
  if (!lowering.graph->value_ids.valid(step.inputs) ||
      !lowering.graph->value_ids.valid(step.outputs)) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  const std::span<const std::uint32_t> inputs =
      lowering.graph->value_ids.view(step.inputs);
  const std::span<const std::uint32_t> outputs =
      lowering.graph->value_ids.view(step.outputs);
  const graph::Node &planned = lowering.layout->nodes[index];
  if (inputs.empty() || outputs.empty() ||
      planned.accesses.size() != step.node.signature.value_count) {
    return Status::fail(Reason::GraphBindingInvalid);
  }

  if (!lowering.cpu()) {
    try {
      lowering.accel_refs.emplace_back();
      auto &refs = lowering.accel_refs.back();
      refs.reserve(planned.accesses.size());
      for (const graph::Access &access : planned.accesses) {
        const kernel::BufferRole role =
            access.mode == resource::AccessMode::Read
                ? kernel::BufferRole::Read
                : kernel::BufferRole::Write;
        const auto buffer =
            resident(lowering, index, access.resource, role, nullptr);
        if (!buffer) {
          return Status::fail(Reason::AccelProgramInvalid);
        }
        refs.push_back(*buffer);
      }
      rund::AccelGraphNode node = step.node;
      node.buffers = refs.data();
      node.buffer_count = refs.size();
      node.barrier_before = lowering.barriers[index] != 0u;
      lowering.accel_nodes.push_back(std::move(node));
    } catch (const std::bad_alloc &) {
      return Status::fail(Reason::GraphCapacity);
    }
    return Status::success();
  }

  try {
    const auto &refs = lowering.cpu_refs.back();
    lowering.cpu_nodes.push_back(kernel::GraphNode{
        .buffers = refs.data(),
        .buffer_count = refs.size(),
        .kind = step.node.kind,
        .primitive_hash_hi = step.node.primitive_hash_hi,
        .primitive_hash_lo = step.node.primitive_hash_lo,
        .element_count = step.node.element_count,
        .control = step.node.control,
    });
    CpuRuntimePrimitive runtime{
        .inputs = std::vector<std::uint32_t>{inputs.begin(), inputs.end()},
        .output = step.output,
        .kind = step.primitive,
        .control = step.control,
    };
    const Status planned_runtime = plan(runtime, step);
    if (!planned_runtime && lowering.runtime) {
      lowering.runtime = planned_runtime;
    }
    lowering.program->cpu_graph->runtime->steps.emplace_back(
        std::move(runtime));
    if (step.primitive != Primitive::Reduce) {
      return Status::success();
    }
    CpuDeviceState *const host = cpu_device(*lowering.graph->device);
    const std::size_t input_count =
        lowering.graph->values[inputs.front() - 1u].count;
    if (host == nullptr ||
        input_count > std::numeric_limits<kernel::u32>::max()) {
      return Status::fail(host == nullptr ? Reason::CpuProgramInvalid
                                          : Reason::ShapeCapacity);
    }
    auto collective = std::make_unique<CpuCollective>();
    collective->tile_plan = kernel::ComputeTileExecutor{
        host->workers.backend,
        host->workers.requested_worker_width,
    };
    const kernel::ComputeTilePrepareResult prepared =
        collective->tile_plan.prepare(static_cast<kernel::u32>(input_count));
    if (!prepared.ok) {
      return Status::fail(
          project_reason(prepared.reason, Reason::TileBackendFailed));
    }
    collective->tile_size = prepared.tile_units;
    collective->tile_count = prepared.tile_count;
    lowering.program->cpu_graph->collectives[index] = std::move(collective);
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_compile
