#include "describe/internal.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Describe() {
  using namespace metal_pipeline_describe_internal;
  if (aggregate_selected) {
    return DescribeMetalAggregate(*this);
  }

  MetalPipelineDescribeCapacity capacity{};
  rund::AccelCheck result = DescribeMetalCapacity(*this, capacity);
  if (!result.ok) {
    return result;
  }
  result = DescribeMetalTemplates(*this, capacity);
  if (!result.ok) {
    return result;
  }
  result = DescribeMetalOccurrences(*this);
  if (!result.ok) {
    return result;
  }
  result = DescribeMetalProfile(*this);
  if (!result.ok) {
    return result;
  }
  return DescribeMetalPacking(*this, capacity);
}

#endif

} // namespace rund::node::accel::detail
