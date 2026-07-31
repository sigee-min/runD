#pragma once

#include "../../compute.hpp"

#include <cstdio>

namespace compute_lifecycle_test {

[[nodiscard]] inline int CheckSuite(rund::Session &session,
                                    const std::array<std::int32_t, 4> &input) {
  if (const int result =
          rund::node::test_contract::CheckComputeJobGate(session, input);
      result != 0) {
    std::fprintf(stderr, "compute lifecycle owner=job-gate result=%d\n",
                 result);
    return result;
  }
  if (const int result =
          rund::node::test_contract::CheckComputeProgramConcurrency(session);
      result != 0) {
    std::fprintf(stderr, "compute lifecycle owner=concurrency result=%d\n",
                 result);
    return result;
  }
  if (const int result =
          rund::node::test_contract::CheckComputeCollectives(session);
      result != 0) {
    std::fprintf(stderr, "compute lifecycle owner=collectives result=%d\n",
                 result);
    return result;
  }
  const int result = rund::node::test_contract::CheckComputeAwait(session);
  if (result != 0) {
    std::fprintf(stderr, "compute lifecycle owner=await result=%d\n", result);
  }
  return result;
}

} // namespace compute_lifecycle_test
