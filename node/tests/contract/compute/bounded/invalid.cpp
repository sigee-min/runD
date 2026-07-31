#include "local.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>

namespace rund_node_bounded_contract {

int CheckInvalidBoundedCount(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::uint64_t, 4u> input{2u, 2u, 2u, 2u};
  auto program = on(rund::node::test_contract::target_for(backend))
                     .map<std::uint64_t>("bounded-count-invalid", input.size(),
                                         [](auto value) { return value; })
                     .expand(
                         MaxItems{1u}, [](auto value) { return value; },
                         [](auto value, auto) { return value; })
                     .compile();
  if (!program) {
    return 2;
  }
  auto job = program->resident(input);
  if (!job) {
    return 3;
  }
  const Status status = job->run();
  return !status && status.error() == "compute_bounded_count_invalid" ? 0 : 4;
}

bool CheckInputSetMapShapeAdmission() {
  using namespace rund::compute;
  auto mismatch = on(Target::cpu())
                      .input<std::int32_t>(4u)
                      .zip_input<std::int32_t>(1u)
                      .map("map-shape-mismatch",
                           [](auto left, auto right) { return left + right; })
                      .compile();
  return !mismatch && mismatch.error() == "compute_zip_shape_mismatch";
}

} // namespace rund_node_bounded_contract
