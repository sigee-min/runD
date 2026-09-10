#pragma once

#include "failure.hpp"
#include <accel/runtime.hpp>
#include <memory>

namespace rund::node::accel::detail {

struct PreparedKernelPipeline {
  std::shared_ptr<void> owner{};
  AccelRunFacts preparation{};
  bool ok = false;
  PreparedPipelineFailure failure{};
};


} // namespace rund::node::accel::detail
