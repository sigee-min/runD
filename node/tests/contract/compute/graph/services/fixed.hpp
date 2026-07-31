#pragma once

#include "model.hpp"

#include <array>
#include <vector>

namespace rund_node_graph_services {

template <class T>
[[nodiscard]] bool
fixed_resource_metadata(const Info &graph, const std::uint32_t resource,
                        const Visibility visibility,
                        const rund::compute::Rounding rounding,
                        const rund::compute::Overflow overflow,
                        const rund::compute::Approximation approximation) {
  if (resource == 0u || resource > graph.resources.size()) {
    return false;
  }
  const auto &value = graph.resources[resource - 1u];
  return value.type == rund::compute::graph::Value::Fixed &&
         value.integer_bits == T::integer_bits &&
         value.fraction_bits == T::fraction_bits &&
         value.rounding == rounding && value.overflow == overflow &&
         value.approximation == approximation &&
         value.visibility == visibility && value.elements == 4u &&
         value.element_bytes == sizeof(T) && value.bytes == 4u * sizeof(T);
}

template <class T>
[[nodiscard]] bool
default_fixed_resource_metadata(const Info &graph, const std::uint32_t resource,
                                const Visibility visibility) {
  return fixed_resource_metadata<T>(
      graph, resource, visibility, rund::compute::Rounding::NearestEven,
      rund::compute::Overflow::Saturate, rund::compute::Approximation::Exact);
}

template <class T>
[[nodiscard]] bool same_raw_values(const std::vector<T> &actual,
                                   const std::span<const T> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < actual.size(); ++index) {
    if (actual[index].raw() != expected[index].raw()) {
      return false;
    }
  }
  return true;
}

template <class T, class Job>
[[nodiscard]] bool run_fixed_job(Job &job, const Backend backend,
                                 const std::span<const T> expected,
                                 ExecutionEvidence &execution) {
  if (!job.run()) {
    return false;
  }
  auto output = job.read();
  const auto stats = job.stats();
  if (!output || !same_raw_values(*output, expected) ||
      stats.backend != backend || stats.graph_hash == 0u ||
      stats.output_hash == 0u) {
    return false;
  }
  execution = {.graph_hash = stats.graph_hash,
               .output_hash = stats.output_hash};
  return true;
}

} // namespace rund_node_graph_services
