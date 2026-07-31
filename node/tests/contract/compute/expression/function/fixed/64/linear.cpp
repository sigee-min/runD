#include "../../../part/function/linear.hpp"
#include "../../../parts.hpp"

namespace rund::node::test_contract::expression {

int run_linear64(Executor &executor, const compute::Backend backend) {
  return offset_error(
      check_linear(executor, backend, helper_input<compute::Fixed<20, 44>>()),
      1100);
}

} // namespace rund::node::test_contract::expression
