#pragma once

#include <accel/graph/node.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>

namespace rund {

struct AccelGraph {
  const AccelGraphNode *nodes = nullptr;
  std::uint64_t node_count = 0u;
  const std::uint64_t *outputs = nullptr;
  std::uint64_t output_count = 0u;
  rund::kernel::ComputeScalar scalar = rund::kernel::ComputeScalar::Lane32;
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::ComputeFixedFormat fixed_format{};
};

} // namespace rund
