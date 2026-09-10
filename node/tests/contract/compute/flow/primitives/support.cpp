#include "local.hpp"

namespace rund_node_test_flow_primitives {

rund::compute::FlowBuilder Target() {
  return rund::compute::on(rund::compute::Target::cpu(2u));
}

} // namespace rund_node_test_flow_primitives
