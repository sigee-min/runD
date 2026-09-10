#include "local.hpp"

#include "graph_pointwise_multi/internal.hpp"

#include "../../../target/selection.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_virtual::product {

int CheckProductGraphPointwiseMulti(const rund::compute::Backend backend,
                                    const bool require_adapter) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    if (require_adapter) {
      std::fprintf(stderr,
                   "Graph multi required adapter backend=%u reason=%u\n",
                   static_cast<unsigned>(backend),
                   static_cast<unsigned>(opened.reason()));
    }
    return !require_adapter && opened.reason() == Reason::AdapterUnavailable
               ? 0
               : 1;
  }
  constexpr std::array<std::size_t, 3u> PageCounts{5u, 9u, 257u};
  for (std::size_t index = 0u; index < PageCounts.size(); ++index) {
    const int result =
        graph_pointwise_multi::check_scale(*opened, backend, PageCounts[index]);
    if (result != 0) {
      return static_cast<int>(10u * (index + 1u)) + result;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
