#include "local.hpp"

#include <cstdint>

namespace rund_node_bounded_contract {

[[nodiscard]] int CheckCpu() {
  using namespace rund::compute;
  if (!CheckInputSetMapShapeAdmission()) {
    return 99;
  }
  Stats single_filter{};
  const int cpu_single = CheckFilterBackend(Target::cpu(1u), &single_filter);
  if (cpu_single != 0) {
    return 100 + cpu_single;
  }
  Stats parallel_filter{};
  const int cpu_parallel =
      CheckFilterBackend(Target::cpu(4u), &parallel_filter);
  if (cpu_parallel != 0) {
    return 120 + cpu_parallel;
  }
  if (single_filter.graph_hash != parallel_filter.graph_hash ||
      single_filter.output_hash != parallel_filter.output_hash) {
    return 140;
  }
  if (const int typed = CheckTypedBoundedMap(Target::cpu(2u)); typed != 0) {
    return 150 + typed;
  }
  if (const int inactive = CheckInactiveTail(Target::cpu(2u)); inactive != 0) {
    return 160 + inactive;
  }
  if (const int rewrite = CheckReduceRewrite(Target::cpu(2u)); rewrite != 0) {
    return 170 + rewrite;
  }
  return 0;
}

} // namespace rund_node_bounded_contract
