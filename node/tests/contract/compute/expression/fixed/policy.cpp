#include "../part/policy.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {

int run_policy(Executor &executor, const compute::Backend backend) {
  using namespace rund::compute;
  if (const int result = check_left_rescale(executor, backend); result != 0)
    return 680 + result;
  if (const int result =
          check_policy_literals<Fixed<16, 16>>(executor, backend);
      result != 0)
    return 685 + result;
  if (const int result =
          check_policy_literals<Fixed<20, 44>>(executor, backend);
      result != 0)
    return 690 + result;
  if (const int result =
          check_widened_ternary<Fixed<16, 16>>(executor, backend);
      result != 0)
    return 695 + result;
  if (const int result =
          check_widened_ternary<Fixed<20, 44>>(executor, backend);
      result != 0)
    return 698 + result;
  return 0;
}

} // namespace rund::node::test_contract::expression
