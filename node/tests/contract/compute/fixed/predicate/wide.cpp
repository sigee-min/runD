#include "wide.hpp"

namespace {

struct StandaloneFixedWidePredicateExecutor final {
  template <class Job>
  [[nodiscard]] rund::compute::Status operator()(Job &job) const {
    return job.run();
  }
};

} // namespace

int RunComputeFixedWidePredicateContract() {
  return rund::node::test_contract::RunFixedWidePredicateInventory(
      StandaloneFixedWidePredicateExecutor{});
}
