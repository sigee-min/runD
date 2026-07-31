#include "flow/contract/local.hpp"

#include "../target/selection.hpp"

int RunComputeFlowContract() {
  const std::span<const rund::compute::Backend> backends =
      rund::node::test_contract::selected_compute_backends();
  if (const int shape = rund_node_flow_contract::CheckShape(); shape != 0) {
    return 90 + shape;
  }
  if (const int device = rund_node_flow_contract::CheckDevice(); device != 0) {
    return 100 + device;
  }
  if (const int backend =
          rund_node_flow_contract::CheckBackendContracts(backends);
      backend != 0) {
    return backend;
  }
  if (const int basic = rund_node_flow_contract::CheckBasic(); basic != 0) {
    return basic;
  }
  return rund_node_flow_contract::CheckParityBackends(backends);
}
