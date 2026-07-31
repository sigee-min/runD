#include "../part/function/approx.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T>
[[nodiscard]] int run_approx(Executor &executor, const compute::Backend backend,
                             const int base) {
  return offset_error(check_approx(executor, backend, helper_input<T>()), base);
}

} // namespace

int run_approx32(Executor &executor, const compute::Backend backend) {
  return run_approx<compute::Fixed<16, 16>>(executor, backend, 1600);
}

int run_approx64(Executor &executor, const compute::Backend backend) {
  return run_approx<compute::Fixed<20, 44>>(executor, backend, 1700);
}

} // namespace rund::node::test_contract::expression
