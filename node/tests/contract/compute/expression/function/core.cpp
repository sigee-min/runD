#include "../part/function/core.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T>
[[nodiscard]] int run_core(Executor &executor, const compute::Backend backend,
                           const int base) {
  return offset_error(check_core(executor, backend, helper_input<T>()), base);
}

} // namespace

int run_core32(Executor &executor, const compute::Backend backend) {
  return run_core<compute::Fixed<16, 16>>(executor, backend, 800);
}

int run_core64(Executor &executor, const compute::Backend backend) {
  return run_core<compute::Fixed<20, 44>>(executor, backend, 900);
}

} // namespace rund::node::test_contract::expression
