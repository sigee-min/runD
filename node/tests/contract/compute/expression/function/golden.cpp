#include "../part/function/golden.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T>
[[nodiscard]] int run_format(Executor &executor, const compute::Backend backend,
                             const int unit_base, const int window_base) {
  if (const int result = check_unit_golden<T>(executor, backend); result != 0) {
    return unit_base + result;
  }
  if (const int result = check_window_golden<T>(executor, backend);
      result != 0) {
    return window_base + result;
  }
  return 0;
}

} // namespace

int run_format32(Executor &executor, const compute::Backend backend) {
  return run_format<compute::Fixed<16, 16>>(executor, backend, 780, 790);
}

int run_format64(Executor &executor, const compute::Backend backend) {
  return run_format<compute::Fixed<20, 44>>(executor, backend, 880, 890);
}

} // namespace rund::node::test_contract::expression
