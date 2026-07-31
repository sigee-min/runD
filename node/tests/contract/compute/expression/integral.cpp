#include "part/integral.hpp"
#include "parts.hpp"

namespace rund::node::test_contract::expression {

int run_integral(Executor &executor, const compute::Backend backend) {
  using namespace rund::compute;
  if (const int result = check_operators<std::int32_t>(executor, backend);
      result != 0)
    return 100 + result;
  if (const int result = check_boundaries<std::int32_t>(executor, backend);
      result != 0)
    return 150 + result;
  if (const int result = check_operators<std::uint32_t>(executor, backend);
      result != 0)
    return 200 + result;
  if (const int result = check_boundaries<std::uint32_t>(executor, backend);
      result != 0)
    return 250 + result;
  if (const int result = check_operators<std::int64_t>(executor, backend);
      result != 0)
    return 300 + result;
  if (const int result = check_boundaries<std::int64_t>(executor, backend);
      result != 0)
    return 350 + result;
  if (const int result = check_operators<std::uint64_t>(executor, backend);
      result != 0)
    return 400 + result;
  if (const int result = check_boundaries<std::uint64_t>(executor, backend);
      result != 0)
    return 450 + result;
  return 0;
}

} // namespace rund::node::test_contract::expression
