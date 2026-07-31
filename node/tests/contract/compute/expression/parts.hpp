#pragma once

#include <rund/compute/backend.hpp>
#include <rund/compute/ops.hpp>
#include <rund/compute/status.hpp>

namespace rund::node::test_contract::expression {

struct Executor final {
  template <class Job>
  [[nodiscard]] compute::Status operator()(Job &job) const {
    return job.run();
  }
};

using Part = int (*)(Executor &, compute::Backend);

[[nodiscard]] constexpr int offset_error(const int result,
                                         const int base) noexcept {
  return result == 0 ? 0 : base + result;
}

[[nodiscard]] int run_integral(Executor &, compute::Backend);
[[nodiscard]] int run_fixed_storage32(Executor &, compute::Backend);
[[nodiscard]] int run_fixed_storage64(Executor &, compute::Backend);
[[nodiscard]] int run_policy(Executor &, compute::Backend);
[[nodiscard]] int run_format32(Executor &, compute::Backend);
[[nodiscard]] int run_format64(Executor &, compute::Backend);
[[nodiscard]] int run_core32(Executor &, compute::Backend);
[[nodiscard]] int run_core64(Executor &, compute::Backend);
[[nodiscard]] int run_linear32(Executor &, compute::Backend);
[[nodiscard]] int run_linear64(Executor &, compute::Backend);
[[nodiscard]] int run_geometry32(Executor &, compute::Backend);
[[nodiscard]] int run_geometry64(Executor &, compute::Backend);
[[nodiscard]] int run_algebra32(Executor &, compute::Backend);
[[nodiscard]] int run_algebra64(Executor &, compute::Backend);
[[nodiscard]] int run_approx32(Executor &, compute::Backend);
[[nodiscard]] int run_approx64(Executor &, compute::Backend);
[[nodiscard]] int run_cardinality(Executor &, compute::Backend);

} // namespace rund::node::test_contract::expression
