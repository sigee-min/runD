#pragma once

#include "local.hpp"

#include <rund/compute/math.hpp>

#include <type_traits>

namespace rund_node_graph_services {

template <class Program>
[[nodiscard]] bool uses_backend(const Program &program,
                                const Backend expected) {
  const auto backend = program.backend();
  return backend && *backend == expected;
}

template <class Cache>
auto build(rund::compute::Device &device, Cache &cache, const char *name,
           const std::int32_t add, const std::size_t count = 4u) {
  return rund::compute::on(device, cache)
      .template map<std::int32_t>(
          name, count,
          rund::compute::capture(
              [](auto value, auto constant) { return value + constant; }, add))
      .compile();
}

} // namespace rund_node_graph_services
