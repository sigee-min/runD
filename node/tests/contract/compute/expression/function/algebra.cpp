#include "../part/function/algebra.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T>
[[nodiscard]] int run_algebra(Executor &executor,
                              const compute::Backend backend, const int base) {
  return offset_error(check_algebra(executor, backend, helper_input<T>()),
                      base);
}

} // namespace

int run_algebra32(Executor &executor, const compute::Backend backend) {
  return run_algebra<compute::Fixed<16, 16>>(executor, backend, 1400);
}

int run_algebra64(Executor &executor, const compute::Backend backend) {
  return run_algebra<compute::Fixed<20, 44>>(executor, backend, 1500);
}

} // namespace rund::node::test_contract::expression
