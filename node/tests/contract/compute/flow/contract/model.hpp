#pragma once

#include "local.hpp"

#include <rund/compute/math.hpp>

#include "../../../target/selection.hpp"

#include <type_traits>
#include <utility>

namespace rund_node_flow_contract {

using rund::node::test_contract::flow_on;

struct ValueField final {};
struct WeightField final {};
struct ScoreField final {};
struct NestedField final {};
struct AlternateValueField final {};
struct AlternateScoreField final {};
template <std::size_t> struct OpField final {};

template <class T> [[nodiscard]] constexpr auto FlowStore(const auto &value) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return rund::compute::quantize<T>(value);
  } else {
    return value;
  }
}

template <class T>
[[nodiscard]] constexpr T FlowDomainValue(const std::int64_t value) noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    static_assert(T::fraction_bits >= 4u);
    return T::from_raw(static_cast<Raw>(value) << (T::fraction_bits - 4u));
  } else {
    return static_cast<T>(value);
  }
}

template <class T>
[[nodiscard]] auto FlowInput(const rund::compute::Backend backend,
                             const char *const name, const std::size_t count) {
  using namespace rund::compute;
  return backend == Backend::Cpu
             ? on(Target::cpu(2u))
                   .template map<T>(
                       name, count,
                       [](auto value) { return FlowStore<T>(value); })
             : flow_on(backend).template map<T>(
                   name, count, [](auto value) { return FlowStore<T>(value); });
}

template <class Job>
[[nodiscard]] bool CheckFlowParity(Job &job,
                                   const rund::compute::Backend backend,
                                   FlowHash &reference) {
  if (!job.run()) {
    return false;
  }
  auto output = job.read_all();
  const rund::compute::Stats stats = job.stats();
  const FlowHash observed{.graph = stats.graph_hash,
                          .output = stats.output_hash};
  const bool same =
      output && stats.backend == backend && observed.graph != 0u &&
      observed.output != 0u &&
      (reference.graph == 0u || reference.graph == observed.graph) &&
      (reference.output == 0u || reference.output == observed.output);
  if (same && reference.graph == 0u) {
    reference = observed;
  }
  return same;
}

} // namespace rund_node_flow_contract
