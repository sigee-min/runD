#include "local.hpp"

#include "../../target/selection.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckBackend(const Backend backend,
                               rund::compute::graph::Fingerprint &fingerprint,
                               std::uint64_t &output_hash,
                               std::uint64_t &state_hash,
                               std::uint64_t &mixed_hash) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  rund::node::test_contract::require_selected_backend(backend);
  auto device = rund::compute::open(rund::compute::Target::cpu(2u));
#else
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend, 2u));
#endif
  if (!device) {
    return 1;
  }
  if (const int repeat = CheckRepeat(*device, backend); repeat != 0) {
    return 50 + repeat;
  }
  if (const int fixed =
          CheckWideFixed(*device, backend, fingerprint, output_hash);
      fixed != 0) {
    return 100 + fixed;
  }
  if (const int memory = CheckMemory(*device); memory != 0) {
    return 175 + memory;
  }
  if (const int surface = CheckSurface(*device); surface != 0) {
    return 200 + surface;
  }
  if (const int views = CheckViews(*device, backend); views != 0) {
    return 250 + views;
  }
  if (const int hazards = CheckHazards(*device); hazards != 0) {
    return 300 + hazards;
  }
  if (const int zero = CheckZeroWork(*device, backend); zero != 0) {
    return 400 + zero;
  }
  if (const int semantic = CheckSemanticStatus(*device, backend);
      semantic != 0) {
    return 500 + semantic;
  }
  if (const int generations = CheckTransactionalGenerations(
          *device, backend, state_hash, mixed_hash);
      generations != 0) {
    return 550 + generations;
  }
  if (const int chunking = CheckVulkanCheckpointChunking(*device, backend);
      chunking != 0) {
    return 575 + chunking;
  }
  if (const int device_loss = CheckNativeDeviceLoss(*device, backend);
      device_loss != 0) {
    return 600 + device_loss;
  }
  if (const int unknown =
          CheckUnknownCompletionProfileIdentity(*device, backend);
      unknown != 0) {
    return 610 + unknown;
  }
  if (const int profile = CheckProfile(*device, backend); profile != 0) {
    return 625 + profile;
  }
  if (backend == Backend::Cpu) {
    if (const int frozen = CheckFrozenCpuMapBindings(*device); frozen != 0) {
      return 650 + frozen;
    }
    if (const int validation = CheckValidationAndIdentity(*device);
        validation != 0) {
      return 700 + validation;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline
