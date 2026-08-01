#include "pipeline/local.hpp"

#include "../target/selection.hpp"

#include <cstdint>
#include <cstdio>

int RunComputePipelineContract() {
  if (const int guard = rund_node_test_pipeline::CheckMetalGuardTransform();
      guard != 0) {
    std::fprintf(stderr, "pipeline Metal guard contract result=%d\n", guard);
    return 800 + guard;
  }
  rund::compute::graph::Fingerprint fingerprint{};
  rund::compute::graph::Fingerprint sealed_repetition_fingerprint{};
  std::uint64_t output_hash = 0u;
  std::uint64_t state_hash = 0u;
  std::uint64_t mixed_hash = 0u;
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const int result = rund_node_test_pipeline::CheckBackend(
        backend, fingerprint, sealed_repetition_fingerprint, output_hash,
        state_hash, mixed_hash);
    if (result != 0) {
      std::fprintf(stderr, "pipeline contract backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + result;
    }
  }
  return fingerprint && sealed_repetition_fingerprint && output_hash != 0u &&
                 state_hash != 0u && mixed_hash != 0u
             ? 0
             : 1;
}
