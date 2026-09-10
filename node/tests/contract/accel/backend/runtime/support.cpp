#include "local.hpp"

#include <node/accel/pick.hpp>
#include <rund/counter.hpp>

namespace node_accel_contract::backend_runtime {

void OverflowCounter(std::uint64_t &counter) noexcept {
  counter = kCounterMaximum - 1u;
  ::rund::detail::counter::Accumulate(counter, 2u);
}

bool CountersEqual(const std::uint64_t expected,
                   const std::initializer_list<std::uint64_t> values) noexcept {
  for (const std::uint64_t value : values) {
    if (value != expected) {
      return false;
    }
  }
  return true;
}

rund::AccelDevice Pick(const rund::AccelApi api) {
  return rund::node::accel::PickAccel(
      node_accel_contract::backend::Policy({api}));
}

} // namespace node_accel_contract::backend_runtime
