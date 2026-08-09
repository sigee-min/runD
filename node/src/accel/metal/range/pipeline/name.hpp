#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "name/function.hpp"

[[nodiscard]] inline std::string RangePipelineKey(const RangeExec &execution) {
  const RangeShape &shape = execution.plan().shape();
  const RangeGpuShape &gpu = execution.shape();
  std::string key = "range.aggregate.";
  const auto append = [&](const std::uint64_t value) {
    key += std::to_string(value);
    key += ".";
  };
  append(static_cast<std::uint8_t>(execution.candidate()));
  append(gpu.width());
  append(gpu.shared_radius_capacity());
  append(static_cast<std::uint8_t>(execution.operation()));
  append(static_cast<std::uint8_t>(execution.domain()));
  append(static_cast<std::uint8_t>(execution.arithmetic_law()));
  append(static_cast<std::uint8_t>(shape.boundary()));
  append(execution.element_bytes());
  append(static_cast<std::uint8_t>(shape.count()));
  key.pop_back();
  return key;
}
#endif

} // namespace rund::node::accel::detail
