#include "../part/core.hpp"
#include "../parts.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T>
[[nodiscard]] int run_core(Executor &executor, const compute::Backend backend,
                           const int operators_base, const int boundaries_base,
                           const int quantize_base) {
  if (const int result = check_operators<T>(executor, backend); result != 0) {
    return operators_base + result;
  }
  if (const int result = check_boundaries<T>(executor, backend); result != 0) {
    return boundaries_base + result;
  }
  if (const int result = check_quantize_modes<T>(executor, backend);
      result != 0) {
    return quantize_base + result;
  }
  return 0;
}

} // namespace

int run_fixed_storage32(Executor &executor, const compute::Backend backend) {
  return run_core<compute::Fixed<16, 16>>(executor, backend, 500, 550, 575);
}

int run_fixed_storage64(Executor &executor, const compute::Backend backend) {
  return run_core<compute::Fixed<20, 44>>(executor, backend, 600, 650, 675);
}

} // namespace rund::node::test_contract::expression
