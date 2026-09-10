#include "local.hpp"

#include "src/compute/virtual/run/sliding/internal.hpp"

namespace rund_node_test_virtual::product {

int CheckProductReturnedFailureBoundary() {
  namespace sliding = rund::compute::detail::sliding_product_detail;
  const auto state = std::make_shared<sliding::SlidingProductRun>();
  if (state == nullptr ||
      !sliding::requires_returned_output_service(*state, 3u)) {
    return 1;
  }
  state->failure =
      rund::compute::Status::fail(rund::compute::Reason::BackendFailed);
  state->failure_coordinate = 4u;
  return sliding::requires_returned_output_service(*state, 3u) &&
                 !sliding::requires_returned_output_service(*state, 4u) &&
                 !sliding::requires_returned_output_service(*state, 5u)
             ? 0
             : 2;
}

} // namespace rund_node_test_virtual::product
