#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "name/function.hpp"

[[nodiscard]] inline std::string RangePipelineKey(const RangeExec &execution) {
  const RangeIdentity identity = execution.source_identity();
  std::string key = "range.aggregate.";
  key += std::to_string(identity.hi);
  key += ".";
  key += std::to_string(identity.lo);
  return key;
}
#endif

} // namespace rund::node::accel::detail
