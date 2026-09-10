#include "product/local.hpp"

#include "../../target/selection.hpp"

#include <cstdio>

int RunComputeVirtualGraphPointwiseLaterContract() {
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    const int result =
        rund_node_test_virtual::product::CheckProductGraphPointwiseMulti(
            backend, true);
    if (result != 0) {
      std::fprintf(stderr,
                   "compute virtual later-input GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + result;
    }
  }
  return 0;
}
