#include "reservation_internal.hpp"

#include "../../plan.hpp"

namespace rund::node::accel::detail::backend_template_plan {

std::uint64_t primitive_pass_count(const KernelExecutionStep &step) noexcept {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    return 1u;
  case rund::kernel::NodeKind::Scan:
    return step.operation.get<operation::Scan>().plan.pass_count;
  case rund::kernel::NodeKind::SegmentedScan:
    return step.operation.get<operation::SegmentedScan>().plan.pass_count;
  case rund::kernel::NodeKind::SegmentedReduce:
    return step.operation.get<operation::SegmentedReduce>().plan.pass_count;
  case rund::kernel::NodeKind::Sort:
    return step.operation.get<operation::Sort>().plan.radix_pass_count;
  case rund::kernel::NodeKind::Compact:
    return step.operation.get<operation::Compact>().plan.pass_count;
  case rund::kernel::NodeKind::Gather:
    return step.operation.get<operation::Gather>().plan.pass_count;
  case rund::kernel::NodeKind::Histogram:
    return step.operation.get<operation::Histogram>().plan.pass_count;
  case rund::kernel::NodeKind::Partition:
    return step.operation.get<operation::Partition>().plan.pass_count;
  case rund::kernel::NodeKind::Reduce:
    return step.operation.get<operation::Reduce>().plan.pass_count;
  case rund::kernel::NodeKind::Scatter:
    return step.operation.get<operation::Scatter>().plan.pass_count;
  case rund::kernel::NodeKind::ScatterReduce:
    return step.operation.get<operation::ScatterReduce>().plan.pass_count;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window:
    return RangePlanFor(step.operation)->stage_count();
  case rund::kernel::NodeKind::Transform:
    return step.operation.get<operation::Transform>().plan.pass_count;
  case rund::kernel::NodeKind::Matrix:
    return step.operation.get<operation::Matrix>().plan.pass_count;
  case rund::kernel::NodeKind::Factor:
    return step.operation.get<operation::Factor>().plan.pass_count;
  case rund::kernel::NodeKind::Solve:
    return step.operation.get<operation::Solve>().plan.pass_count;
  case rund::kernel::NodeKind::Spectrum:
    return step.operation.get<operation::Spectrum>().plan.pass_count;
  }
  return 0u;
}

} // namespace rund::node::accel::detail::backend_template_plan
