#include "internal.hpp"

#include "../../bindings.hpp"
#include "../../bindings/local.hpp"
#include "../../step/map/local.hpp"

#include <utility>

namespace rund::node::accel::detail::backend_run_detail {

bool bind_primitive(const KernelExecutionStep &step, const RunBinds &run_binds,
                    BoundBindings &out) {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    StepBinds bindings{};
    if (!BuildStepBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Scan: {
    ScanBinds bindings{};
    if (!BuildScanBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Compact: {
    CompactBinds bindings{};
    if (!BuildCompactBinds(step, run_binds, bindings)) {
      return false;
    }
    out = bindings;
    return true;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    SegmentedScanBinds bindings{};
    if (!BuildSegmentedScanBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    SegmentedReduceBinds bindings{};
    if (!BuildSegmentedReduceBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Sort: {
    SortBinds bindings{};
    if (!BuildSortBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Gather: {
    GatherBinds bindings{};
    if (!BuildGatherBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Histogram: {
    HistogramBinds bindings{};
    if (!BuildHistogramBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Partition: {
    PartitionBinds bindings{};
    if (!BuildPartitionBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Reduce: {
    ReduceBinds bindings{};
    if (!BuildReduceBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Scatter: {
    ScatterBinds bindings{};
    if (!BuildScatterBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    ScatterReduceBinds bindings{};
    if (!BuildScatterReduceBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window: {
    RangeBinds bindings{};
    if (!BuildRangeBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Transform: {
    TransformBinds bindings{};
    if (!BuildTransformBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Matrix: {
    MatrixBinds bindings{};
    if (!BuildMatrixBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Factor: {
    FactorBinds bindings{};
    if (!BuildFactorBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Solve: {
    SolveBinds bindings{};
    if (!BuildSolveBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Spectrum: {
    SpectrumBinds bindings{};
    if (!BuildSpectrumBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  default:
    return false;
  }
}

bool bind_control(const KernelExecutionStep &step, const RunBinds &run_binds,
                  BoundControl &out) noexcept {
  out = BoundControl{.control = step.control};
  if (!step.control.valid(step.graph_binding_indices.size())) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  const auto bind = [&](const std::uint32_t local,
                        const rund::kernel::ResidentBufferRef *&ref,
                        const std::shared_ptr<void> *&handle) {
    return local < step.graph_binding_indices.size() &&
           ReadBinding(source, step.graph_binding_indices[local], ref,
                       handle) &&
           BindingReady(handle);
  };
  return (!step.control.has_count() ||
          bind(step.control.count_binding, out.count, out.count_handle)) &&
         (!step.control.has_predicate() ||
          bind(step.control.predicate_binding, out.predicate,
               out.predicate_handle));
}

} // namespace rund::node::accel::detail::backend_run_detail

namespace rund::node::accel::detail {

bool RebindBoundStep(const BoundStep &source, const RunBinds &binds,
                     BoundStep &out) {
  if (source.step == nullptr || source.planned == nullptr) {
    return false;
  }
  out = BoundStep{.index = source.index,
                  .step = source.step,
                  .planned = source.planned,
                  .source_binds = &binds,
                  .barrier_before = source.barrier_before};
  if (!backend_run_detail::bind_primitive(*source.step, binds, out.bindings) ||
      !backend_run_detail::bind_control(*source.step, binds, out.control)) {
    return false;
  }
  // Dense View lowering changes bindings, not the frozen dispatch frontier.
  if (source.step->kind() == rund::kernel::NodeKind::Map) {
    out.map_windows = source.map_windows;
  }
  return true;
}

rund::kernel::BindingSet MapBindingFor(const BoundStep &step) noexcept {
  if (step.step == nullptr || step.planned == nullptr ||
      step.step->kind() != rund::kernel::NodeKind::Map) {
    return {};
  }
  const StepBinds *const bindings = BindingsFor<StepBinds>(step);
  return bindings == nullptr
             ? rund::kernel::BindingSet{}
             : BindMapStep(*step.step, *step.planned, *bindings);
}

} // namespace rund::node::accel::detail
