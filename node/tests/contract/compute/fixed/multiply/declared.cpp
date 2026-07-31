#include "declared.hpp"

namespace {

struct StandaloneFixedDeclaredMultiplyExecutor final {
  template <class Job>
  [[nodiscard]] rund::compute::Status operator()(Job &job) const {
    return job.run();
  }
};

} // namespace

int RunComputeFixedDeclaredMultiplyContract() {
  return rund::node::test_contract::RunFixedDeclaredMultiplyInventory(
      StandaloneFixedDeclaredMultiplyExecutor{});
}
