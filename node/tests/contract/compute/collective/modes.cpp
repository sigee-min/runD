#include "modes/local.hpp"

#include "../../target/selection.hpp"

int RunComputeCollectiveModesContract() {
  rund_node_collective_modes::Evidence evidence{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!rund_node_collective_modes::CheckBackend(backend, evidence)) {
      return 10 + static_cast<int>(backend);
    }
  }
  return 0;
}
