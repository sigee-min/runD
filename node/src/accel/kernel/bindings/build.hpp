#pragma once

#include "compact.hpp"
#include "factor.hpp"
#include "gather.hpp"
#include "histogram.hpp"
#include "matrix.hpp"
#include "partition.hpp"
#include "reduce.hpp"
#include "scan.hpp"
#include "scatter.hpp"
#include "segmented.hpp"
#include "solve.hpp"
#include "sort.hpp"
#include "spectrum.hpp"
#include "stencil.hpp"
#include "step.hpp"
#include "transform.hpp"

namespace rund::node::accel::detail {

struct KernelExecutionStep;

[[nodiscard]] bool BuildStepBinds(const KernelExecutionStep &step,
                                  const RunBinds &run_binds, StepBinds &out);

[[nodiscard]] bool BuildScanBinds(const KernelExecutionStep &step,
                                  const RunBinds &run_binds, ScanBinds &out);

[[nodiscard]] bool BuildCompactBinds(const KernelExecutionStep &step,
                                     const RunBinds &run_binds,
                                     CompactBinds &out);

[[nodiscard]] bool BuildSegmentedScanBinds(const KernelExecutionStep &step,
                                           const RunBinds &run_binds,
                                           SegmentedScanBinds &out);

[[nodiscard]] bool BuildSegmentedReduceBinds(const KernelExecutionStep &step,
                                             const RunBinds &run_binds,
                                             SegmentedReduceBinds &out);

[[nodiscard]] bool BuildSortBinds(const KernelExecutionStep &step,
                                  const RunBinds &run_binds, SortBinds &out);

[[nodiscard]] bool BuildGatherBinds(const KernelExecutionStep &step,
                                    const RunBinds &run_binds,
                                    GatherBinds &out);

[[nodiscard]] bool BuildHistogramBinds(const KernelExecutionStep &step,
                                       const RunBinds &run_binds,
                                       HistogramBinds &out);

[[nodiscard]] bool BuildPartitionBinds(const KernelExecutionStep &step,
                                       const RunBinds &run_binds,
                                       PartitionBinds &out);

[[nodiscard]] bool BuildScatterBinds(const KernelExecutionStep &step,
                                     const RunBinds &run_binds,
                                     ScatterBinds &out);
[[nodiscard]] bool BuildScatterReduceBinds(const KernelExecutionStep &step,
                                           const RunBinds &run_binds,
                                           ScatterReduceBinds &out);

[[nodiscard]] bool BuildStencilBinds(const KernelExecutionStep &step,
                                     const RunBinds &run_binds,
                                     StencilBinds &out);

[[nodiscard]] bool BuildTransformBinds(const KernelExecutionStep &step,
                                       const RunBinds &run_binds,
                                       TransformBinds &out);

[[nodiscard]] bool BuildMatrixBinds(const KernelExecutionStep &step,
                                    const RunBinds &run_binds,
                                    MatrixBinds &out);

[[nodiscard]] bool BuildFactorBinds(const KernelExecutionStep &step,
                                    const RunBinds &run_binds,
                                    FactorBinds &out);

[[nodiscard]] bool BuildSolveBinds(const KernelExecutionStep &step,
                                   const RunBinds &run_binds, SolveBinds &out);

[[nodiscard]] bool BuildSpectrumBinds(const KernelExecutionStep &step,
                                      const RunBinds &run_binds,
                                      SpectrumBinds &out);

[[nodiscard]] bool BuildReduceBinds(const KernelExecutionStep &step,
                                    const RunBinds &run_binds,
                                    ReduceBinds &out);

} // namespace rund::node::accel::detail
