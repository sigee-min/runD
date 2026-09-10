#include "local.hpp"

#include "graph_pointwise/internal.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>

namespace rund_node_test_virtual::product {

int CheckProductGraphPointwise(const rund::compute::Backend backend) {
  auto opened =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == rund::compute::Reason::AdapterUnavailable ? 0 : 1;
  }
  if (const int result = graph_pointwise::check_recovery(*opened, backend);
      result != 0) {
    return 10 + result;
  }
  if (backend == rund::compute::Backend::Cpu) {
    return 0;
  }
  if (const int result = graph_pointwise::check_scale(*opened, backend, 9u);
      result != 0) {
    return 20 + result;
  }
  if (const int result = graph_pointwise::check_scale(*opened, backend, 257u);
      result != 0) {
    return 30 + result;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
