#include "local.hpp"

#include "../../backend.hpp"
#include "../../cpu/graph.hpp"
#include "../../fixed/format.hpp"
#include "../../type.hpp"
#include "../local.hpp"
#include "../scan.hpp"

#include <accel/graph/factory.hpp>
#include <kernel/program/compute/scan/identity.hpp>

#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace rund::compute::detail::graph_compile {
namespace {

[[nodiscard]] Result<kernel::GraphControl> control(const GraphState &graph,
                                                   const ScanStep &scan) {
  kernel::GraphControl lowered{};
  if (scan.control.empty() && scan.control.iteration == 0u) {
    return Result<kernel::GraphControl>::success(lowered);
  }
  if (scan.count == 0u || scan.control.count != scan.count ||
      scan.control.predicate != 0u ||
      scan.control.capacity != graph.values[scan.input - 1u].count ||
      scan.control.iteration == 0u) {
    return Result<kernel::GraphControl>::fail(Reason::BoundedCountInvalid);
  }
  lowered = kernel::GraphControl{
      .count_source = graph.values[scan.count - 1u].type == Type::U64
                          ? kernel::GraphControlSource::U64
                          : kernel::GraphControlSource::U32,
      .count_binding = 1u,
      .capacity = scan.control.capacity,
      .iteration = scan.control.iteration,
  };
  return Result<kernel::GraphControl>::success(lowered);
}

} // namespace

Status scan(Lowering &lowering, const std::size_t index, const ScanStep &step) {
  const auto operation = graph_detail::scan_operation(step.operation);
  if (!operation) {
    return Status::fail(Reason::ScanOpUnsupported);
  }
  const auto element =
      graph_detail::scan_element(lowering.graph->values[step.input - 1u].type);
  if (!element) {
    return Status::fail(Reason::TypeUnsupported);
  }
  auto execution = control(*lowering.graph, step);
  if (!execution) {
    return Status::fail(execution.reason());
  }
  const std::size_t input_count = lowering.graph->values[step.input - 1u].count;
  const kernel::ScanDesc desc{
      .op = *operation,
      .element = *element,
      .element_count = input_count,
      .block_size = graph_detail::collective_block(input_count),
      .count_source =
          step.count == 0u
              ? kernel::ComputeCountSource::Descriptor
              : (type_bytes(lowering.graph->values[step.count - 1u].type) == 8u
                     ? kernel::ComputeCountSource::BufferU64
                     : kernel::ComputeCountSource::BufferU32),
  };
  if (lowering.cpu()) {
    CpuDeviceState *const host = cpu_device(*lowering.graph->device);
    if (host == nullptr || !execution->valid(lowering.cpu_refs.back().size()) ||
        input_count > std::numeric_limits<kernel::u32>::max()) {
      return Status::fail(host == nullptr ? Reason::CpuProgramInvalid
                          : input_count >
                                  std::numeric_limits<kernel::u32>::max()
                              ? Reason::ShapeCapacity
                              : Reason::GraphBindingInvalid);
    }
    try {
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
      collective->needs_prefixes = true;
      lowering.program->cpu_graph->collectives[index] = std::move(collective);
      const auto &refs = lowering.cpu_refs.back();
      const kernel::ScanHash hash = kernel::HashScan(desc);
      lowering.cpu_nodes.push_back(kernel::GraphNode{
          .buffers = refs.data(),
          .buffer_count = refs.size(),
          .kind = kernel::NodeKind::Scan,
          .primitive_hash_hi = hash.hi,
          .primitive_hash_lo = hash.lo,
          .element_count = desc.element_count,
          .control = *execution,
      });
      lowering.program->cpu_graph->runtime->steps.emplace_back(CpuRuntimeScan{
          .input = step.input,
          .output = step.output,
          .count = step.count,
          .operation = step.operation,
          .control = step.control,
      });
    } catch (const std::bad_alloc &) {
      return Status::fail(Reason::GraphCapacity);
    }
    return Status::success();
  }

  const auto input =
      resident(lowering, index, step.input, kernel::BufferRole::Read, "input");
  const auto output = resident(lowering, index, step.output,
                               kernel::BufferRole::Write, "output");
  const auto count = step.count == 0u
                         ? std::optional<rund::AccelGraphBufferRef>{}
                         : resident(lowering, index, step.count,
                                    kernel::BufferRole::Read, "logical_count");
  if (!input || !output || (step.count != 0u && !count)) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  try {
    lowering.accel_refs.push_back({*input});
    auto &refs = lowering.accel_refs.back();
    if (count) {
      refs.push_back(*count);
    }
    refs.push_back(*output);
    if (!execution->valid(refs.size())) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    auto node = rund::AccelScan(refs.data(), refs.size(), desc);
    node.control = *execution;
    node.barrier_before = lowering.barriers[index] != 0u;
    lowering.accel_nodes.push_back(std::move(node));
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_compile
