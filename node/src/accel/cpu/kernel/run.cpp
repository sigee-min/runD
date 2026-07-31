#include "run.hpp"

#include "../../kernel/step/map/local.hpp"
#include "../buffer.hpp"
#include "../collective.hpp"

#include <rund/counter.hpp>

#include <cstring>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelCheck Invalid() noexcept {
  return rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

[[nodiscard]] rund::AccelCheck
Reset(const BackendRun &run, const std::size_t step, std::size_t &cursor) {
  if (run.resets == nullptr || run.resets->empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (run.pick == nullptr) {
    return Invalid();
  }
  if (cursor < run.resets->size() && (*run.resets)[cursor].step.index < step) {
    return Invalid();
  }
  std::uint64_t bytes = 0u;
  const std::size_t begin = cursor;
  while (cursor < run.resets->size() &&
         (*run.resets)[cursor].step.index == step) {
    const BoundReset &reset = (*run.resets)[cursor];
    CpuBufferResult target = LookupCpuResidentView(
        *run.pick, reset.ref, reset.handle, rund::kernel::kResidentUsageWrite);
    if (!target.check.ok || target.buffer == nullptr) {
      return target.check.ok
                 ? rund::AccelCheck{false, "accel_kernel_reset_invalid"}
                 : target.check;
    }
    std::vector<std::uint8_t> &data = target.buffer->data;
    if (reset.ref.stride_bytes == reset.ref.element_bytes) {
      std::memset(
          data.data() + reset.ref.offset_bytes, 0,
          static_cast<std::size_t>(reset.ref.count * reset.ref.element_bytes));
    } else {
      for (std::uint64_t ordinal = 0u; ordinal < reset.ref.count; ++ordinal) {
        std::memset(data.data() + reset.ref.offset_bytes +
                        ordinal * reset.ref.stride_bytes,
                    0, static_cast<std::size_t>(reset.ref.element_bytes));
      }
    }
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, reset.ref.count * reset.ref.element_bytes);
    ++cursor;
  }
  if (cursor == begin) {
    return rund::AccelCheck{true, "ok"};
  }
  CpuAdapter *const adapter = CpuAdapterFromPick(*run.pick);
  if (adapter == nullptr) {
    return Invalid();
  }
  {
    std::lock_guard lock{adapter->mutex};
    adapter->reset_command_count = ::rund::detail::counter::SaturatingAdd(
        adapter->reset_command_count,
        static_cast<std::uint64_t>(cursor - begin));
    adapter->reset_bytes =
        ::rund::detail::counter::SaturatingAdd(adapter->reset_bytes, bytes);
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck RunStep(const BackendRun &run,
                                       const BoundStep &bound) {
  if (run.pick == nullptr || bound.step == nullptr ||
      bound.planned == nullptr) {
    return Invalid();
  }
  const rund::AccelDevice &pick = *run.pick;
  const KernelExecutionStep &step = *bound.step;
  const PlannedStep &planned = *bound.planned;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const rund::kernel::BindingSet binding = MapBindingFor(bound);
    return ExecuteCpuMapStep(pick, planned, bound.map_windows, binding);
  }
  case rund::kernel::NodeKind::Scan: {
    const auto &active = step.operation.get<operation::Scan>();
    const ScanBinds *const source = BindingsFor<ScanBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuScan(pick, active.plan, planned.domain, *source);
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &active = step.operation.get<operation::SegmentedScan>();
    const SegmentedScanBinds *const source =
        BindingsFor<SegmentedScanBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuSegmentedScan(pick, active.desc, active.plan,
                                   planned.domain, *source);
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto &active = step.operation.get<operation::SegmentedReduce>();
    const SegmentedReduceBinds *const source =
        BindingsFor<SegmentedReduceBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuSegmentedReduce(pick, active.desc, active.plan,
                                     planned.domain, *source);
  }
  case rund::kernel::NodeKind::Sort: {
    const auto &active = step.operation.get<operation::Sort>();
    const SortBinds *const source = BindingsFor<SortBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuSort(pick, active.desc, active.plan, planned.domain,
                          *source);
  }
  case rund::kernel::NodeKind::Compact: {
    const auto &active = step.operation.get<operation::Compact>();
    const CompactBinds *const source = BindingsFor<CompactBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuCompact(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Gather: {
    const auto &active = step.operation.get<operation::Gather>();
    const GatherBinds *const source = BindingsFor<GatherBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuGather(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Histogram: {
    const auto &active = step.operation.get<operation::Histogram>();
    const HistogramBinds *const source = BindingsFor<HistogramBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuHistogram(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Partition: {
    const auto &active = step.operation.get<operation::Partition>();
    const PartitionBinds *const source = BindingsFor<PartitionBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuPartition(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto &active = step.operation.get<operation::Reduce>();
    const ReduceBinds *const source = BindingsFor<ReduceBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuReduce(pick, active.desc, active.plan, planned.domain,
                            *source);
  }
  case rund::kernel::NodeKind::Scatter: {
    const auto &active = step.operation.get<operation::Scatter>();
    const ScatterBinds *const source = BindingsFor<ScatterBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuScatter(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto &active = step.operation.get<operation::ScatterReduce>();
    const auto *source = BindingsFor<ScatterReduceBinds>(bound);
    return source == nullptr
               ? Invalid()
               : ExecuteCpuScatterReduce(pick, active.plan, *source);
  }
  case rund::kernel::NodeKind::Stencil: {
    const auto &active = step.operation.get<operation::Stencil>();
    const StencilBinds *const source = BindingsFor<StencilBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuStencil(pick, active.desc, active.plan, planned.domain,
                             *source);
  }
  case rund::kernel::NodeKind::Transform: {
    const auto &active = step.operation.get<operation::Transform>();
    const TransformBinds *const source = BindingsFor<TransformBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuTransform(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Matrix: {
    const auto &active = step.operation.get<operation::Matrix>();
    const MatrixBinds *const source = BindingsFor<MatrixBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuMatrix(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Factor: {
    const auto &active = step.operation.get<operation::Factor>();
    const FactorBinds *const source = BindingsFor<FactorBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuFactor(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Solve: {
    const auto &active = step.operation.get<operation::Solve>();
    const SolveBinds *const source = BindingsFor<SolveBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuSolve(pick, active.desc, active.plan, *source);
  }
  case rund::kernel::NodeKind::Spectrum: {
    const auto &active = step.operation.get<operation::Spectrum>();
    const SpectrumBinds *const source = BindingsFor<SpectrumBinds>(bound);
    if (source == nullptr) {
      return Invalid();
    }
    return ExecuteCpuSpectrum(pick, active.desc, active.plan, *source);
  }
  default:
    return Invalid();
  }
}

} // namespace

rund::AccelCheck RunCpuKernel(const BackendRun &run) {
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return Invalid();
  }
  std::size_t reset_cursor = 0u;
  std::uint64_t failed_batches = 0u;
  std::uint64_t first_failed_batch = 0u;
  std::uint32_t first_status = 0u;
  for (std::size_t index = 0u; index < run.step_count; ++index) {
    const rund::AccelCheck reset =
        Reset(run, run.steps[index].index, reset_cursor);
    if (!reset.ok) {
      return reset;
    }
    const rund::AccelCheck step = RunStep(run, run.steps[index]);
    if (step.failed_batches != 0u) {
      if (failed_batches == 0u) {
        first_failed_batch = step.first_failed_batch;
        first_status = step.first_status;
      }
      failed_batches += step.failed_batches;
    }
    if (!step.ok) {
      return step;
    }
  }
  if (run.resets != nullptr && reset_cursor != run.resets->size()) {
    return Invalid();
  }
  rund::AccelCheck result{true, "ok"};
  result.failed_batches = failed_batches;
  result.first_failed_batch = first_failed_batch;
  result.first_status = first_status;
  return result;
}

rund::AccelCheck PrepareCpuKernel(const BackendRun &run,
                                  std::shared_ptr<void> &prepared,
                                  PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  return run.pick != nullptr && run.steps != nullptr && run.step_count != 0u
             ? rund::AccelCheck{true, "ok"}
             : Invalid();
}

rund::AccelCheck
SubmitPreparedCpuKernel(const BackendRun &run, const std::shared_ptr<void> &,
                        const KernelCompletion completion, void *const user,
                        PreparedMemoryMeter *,
                        const std::shared_ptr<void> &) noexcept {
  if (completion == nullptr) {
    return Invalid();
  }
  completion(user, KernelResult{
                       .check = RunCpuKernel(run),
                       .stats = {.dispatch_count = run.final_dispatch_count,
                                 .ok = true,
                                 .reason = "ok"},
                   });
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
