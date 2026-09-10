#include "internal.hpp"

namespace rund_node_test_virtual::product::graph_resident_host {

rund::compute::Result<Program>
build_program(const rund::compute::Device &device) {
  return Workload::build(device);
}

bool validate_program(const Program &program) noexcept {
  return Workload::validate(program);
}

} // namespace rund_node_test_virtual::product::graph_resident_host
