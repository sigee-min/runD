#include "support.hpp"

#include <array>
#include <cstdio>
#include <variant>

namespace rund_node_memory_contract::cpu_primitive {

int CheckEmpty() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;

  CpuGraphRun empty_run{};
  node_compute_allocation::Start();
  CpuPrimitiveScratch *const empty_scratch = &cpu_step_scratch(empty_run, 0u);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      empty_scratch != &empty_run.empty_scratch ||
      !std::holds_alternative<std::monostate>(*empty_scratch) ||
      empty_run.storage != nullptr) {
    return 1;
  }

  const std::array<CpuRuntimePrimitive, 9u> no_scratch{
      ScratchPrimitive(Primitive::SegmentedScan,
                       PlanSegmentedScan(SegmentedScanDesc{
                           .op = SegmentedScanOp::InclusiveSum,
                           .element = SegmentedScanElement::U32,
                           .element_count = 8u,
                           .block_size = 4u,
                       })),
      ScratchPrimitive(Primitive::SegmentedReduce,
                       PlanSegmentedReduce(SegmentedReduceDesc{
                           .op = ReduceOp::Sum,
                           .element = ReduceElement::U32,
                           .element_count = 8u,
                           .block_size = 4u,
                       })),
      ScratchPrimitive(Primitive::Compact, PlanCompact(CompactDesc{
                                               .element_count = 8u,
                                               .output_capacity = 8u,
                                               .flag_bytes = 4u,
                                               .output_bytes = 4u,
                                           })),
      ScratchPrimitive(Primitive::Gather, PlanGather(GatherDesc{
                                              .element = GatherElement::U32,
                                              .element_count = 8u,
                                              .source_count = 16u,
                                          })),
      ScratchPrimitive(Primitive::Histogram, PlanHistogram(HistogramDesc{
                                                 .index = HistogramIndex::U32,
                                                 .count = HistogramCount::U32,
                                                 .element_count = 8u,
                                                 .bin_count = 4u,
                                             })),
      ScratchPrimitive(Primitive::Partition, PlanPartition(PartitionDesc{
                                                 .element_count = 8u,
                                                 .flag_bytes = 4u,
                                                 .value_bytes = 4u,
                                             })),
      ScratchPrimitive(Primitive::Reduce, PlanReduce(ReduceDesc{
                                              .op = ReduceOp::Sum,
                                              .element = ReduceElement::U32,
                                              .element_count = 8u,
                                              .block_size = 4u,
                                          })),
      ScratchPrimitive(Primitive::Stencil,
                       PlanStencil(StencilDesc{
                           .op = StencilOp::Sum,
                           .element = StencilElement::U32,
                           .boundary = StencilBoundary::Clamp,
                           .element_count = 8u,
                           .radius = 1u,
                       })),
      ScratchPrimitive(Primitive::Matrix,
                       PlanMatrix(MatrixDesc{
                           .op = rund::kernel::MatrixOp::Mul,
                           .layout = MatrixLayout::RowMajor,
                           .arithmetic = MatrixArithmetic::Fixed,
                           .rows = 2u,
                           .cols = 2u,
                           .inner = 2u,
                           .batch_count = 3u,
                           .element_bytes = 4u,
                           .fixed_format = ScratchFixedFormat(4u),
                       })),
  };
  for (std::size_t index = 0u; index < no_scratch.size(); ++index) {
    const CpuRuntimePrimitive &primitive = no_scratch[index];
    const bool valid =
        std::visit([](const auto &plan) { return plan.ok; }, primitive.plan);
    if (!valid) {
      const char *const reason = std::visit(
          [](const auto &plan) { return plan.reason; }, primitive.plan);
      std::fprintf(stderr, "cpu scratch plan invalid index=%zu reason=%s\n",
                   index, reason);
      return 2;
    }
    if (!CheckNoCpuPrimitiveScratch(primitive)) {
      std::fprintf(stderr, "cpu scratch unexpectedly retained index=%zu\n",
                   index);
      return 2;
    }
  }
  return 0;
}

} // namespace rund_node_memory_contract::cpu_primitive
