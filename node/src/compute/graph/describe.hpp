#pragma once

#include "state.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <rund/compute/graph/info.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail::graph_detail {

struct Description final {
  graph::Info info{};
  std::vector<compute_dsl::ComputeOp> map_operations{};
  Status status{Status::success()};
};

[[nodiscard]] Description describe(const std::shared_ptr<GraphState> &state);

} // namespace rund::compute::detail::graph_detail
