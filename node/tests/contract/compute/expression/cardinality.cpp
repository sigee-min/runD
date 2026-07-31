#include "part/cardinality.hpp"
#include "parts.hpp"

namespace rund::node::test_contract::expression {

int run_cardinality(Executor &executor, const compute::Backend backend) {
  using namespace rund::compute;
  if (const int result = check_cardinality<std::int32_t>(executor, backend);
      result != 0)
    return 700 + result;
  if (const int result = check_cardinality<std::uint32_t>(executor, backend);
      result != 0)
    return 710 + result;
  if (const int result = check_cardinality<std::int64_t>(executor, backend);
      result != 0)
    return 720 + result;
  if (const int result = check_cardinality<std::uint64_t>(executor, backend);
      result != 0)
    return 730 + result;
  if (const int result = check_cardinality<Fixed<16, 16>>(executor, backend);
      result != 0)
    return 740 + result;
  if (const int result = check_cardinality<Fixed<20, 44>>(executor, backend);
      result != 0)
    return 750 + result;
  if (const int result = check_golden<Fixed<16, 16>>(executor, backend);
      result != 0)
    return 760 + result;
  if (const int result = check_golden<Fixed<20, 44>>(executor, backend);
      result != 0)
    return 770 + result;
  return 0;
}

} // namespace rund::node::test_contract::expression
