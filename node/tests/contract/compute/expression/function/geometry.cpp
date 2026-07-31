#include "../part/function/geometry.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T>
[[nodiscard]] int run_geometry(Executor &executor,
                               const compute::Backend backend, const int base) {
  return offset_error(check_geometry(executor, backend, helper_input<T>()),
                      base);
}

} // namespace

int run_geometry32(Executor &executor, const compute::Backend backend) {
  return run_geometry<compute::Fixed<16, 16>>(executor, backend, 1200);
}

int run_geometry64(Executor &executor, const compute::Backend backend) {
  return run_geometry<compute::Fixed<20, 44>>(executor, backend, 1300);
}

} // namespace rund::node::test_contract::expression
