#include "parts.hpp"

#include "../../target/selection.hpp"

#include <array>

int RunComputeExpressionsContract() {
  using namespace rund::compute;
  using namespace rund::node::test_contract::expression;

  constexpr std::array<Part, 17u> parts{
      run_integral,    run_fixed_storage32, run_fixed_storage64, run_policy,
      run_format32,    run_core32,          run_linear32,        run_geometry32,
      run_algebra32,   run_approx32,        run_format64,        run_core64,
      run_linear64,    run_geometry64,      run_algebra64,       run_approx64,
      run_cardinality,
  };

  Executor executor;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    for (const Part part : parts) {
      if (const int result = part(executor, backend); result != 0) {
        return 1000 * static_cast<int>(backend) + result;
      }
    }
  }
  return 0;
}
