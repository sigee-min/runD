#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"
#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <cstdio>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::binary_test {
namespace {

using rund::compute::Backend;
using rund::compute::Reason;
using rund::compute::ResidencyConfig;
using rund::compute::Status;
using rund::compute::VirtualBacking;
using rund::compute::detail::VirtualBackingAccess;
using rund::compute::detail::VirtualPipelineAccess;
using rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] std::uint64_t version(VirtualBacking &backing) noexcept {
  std::lock_guard lock{VirtualBackingAccess::gate(backing)};
  return VirtualBackingAccess::version(backing);
}

[[nodiscard]] bool exact_run(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    VirtualBacking &output, const std::span<const std::uint32_t> expected,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t queue_expected, const std::uint64_t observations,
    const std::uint64_t reuses, const std::uint64_t warm_rearms,
    std::shared_ptr<DeviceVsmProductOwner> &exact_owner) noexcept {
  rund_node_test_device_vsm_product::RouteObservation observation{};
  const Status status =
      rund_node_test_device_vsm_product::RunThroughDeviceVsmProductRoute(
          state, observation);
  std::uint64_t queue = 0u;
  std::vector<std::uint32_t> actual(expected.size());
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(observation.owner);
  const std::uint64_t bytes = expected.size_bytes();
  const bool valid =
      status && queue_counter != nullptr && queue_counter(state, queue) &&
      queue == queue_expected &&
      output.read(0u, std::as_writable_bytes(std::span{actual})) &&
      std::equal(actual.begin(), actual.end(), expected.begin(),
                 expected.end()) &&
      owner != nullptr && (exact_owner == nullptr || owner == exact_owner) &&
      owner->output_hash_valid &&
      ExactDeviceVsmEvidence(observation, 5u, 5u, bytes, bytes, 0u) &&
      observation.evidence.cold_prepare_count == 1u &&
      observation.evidence.warm_rearm_count == warm_rearms &&
      observation.evidence.output_hash_observation_count == observations &&
      observation.evidence.output_hash_reuse_count == reuses;
  if (exact_owner == nullptr) {
    exact_owner = owner;
  }
  return valid;
}

} // namespace

bool RunResidentHashCacheCase(
    const Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    bool &unavailable) noexcept {
  using namespace rund::compute;
  constexpr std::uint64_t pages = 5u;
  const std::uint64_t elements = pages * PageElements - 3u;
  unavailable = false;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    unavailable = device.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program =
      on(*device)
          .map<std::uint32_t>("device-vsm-resident-hash-cache", PageElements,
                              [](auto value) { return value * 5u + 9u; })
          .compile();
  auto input_backing =
      resident_virtual_backing<std::uint32_t>(*device, elements);
  auto output_backing =
      resident_virtual_backing<std::uint32_t>(*device, elements);
  std::vector<std::uint32_t> input(static_cast<std::size_t>(elements));
  std::vector<std::uint32_t> expected(input.size());
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::uint32_t>(index * 11u + 3u);
    expected[index] = input[index] * 5u + 9u;
  }
  if (!program || !input_backing || !output_backing ||
      !(*input_backing)->write(0u, std::as_bytes(std::span{input}))) {
    return false;
  }
  auto input_buffer = virtual_buffer<std::uint32_t>(elements, *input_backing);
  auto output_buffer = virtual_buffer<std::uint32_t>(elements, *output_backing);
  auto pipeline =
      input_buffer && output_buffer
          ? virtual_pipeline(*program, *input_buffer, *output_buffer,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!pipeline || queue_counter == nullptr) {
    return false;
  }
  const auto state = VirtualPipelineAccess::state(*pipeline);
  std::uint64_t queue_before = 0u;
  const std::uint64_t output_version_before = version(**output_backing);
  std::shared_ptr<DeviceVsmProductOwner> owner{};
  const bool cold = state != nullptr && queue_counter(state, queue_before) &&
                    exact_run(state, **output_backing, expected, queue_counter,
                              queue_before + 1u, 1u, 0u, 0u, owner);
  const bool warm =
      cold && exact_run(state, **output_backing, expected, queue_counter,
                        queue_before + 2u, 1u, 1u, 1u, owner);
  const bool invalidated = warm && (*input_backing)->invalidate();
  const bool refreshed =
      invalidated && exact_run(state, **output_backing, expected, queue_counter,
                               queue_before + 3u, 2u, 1u, 2u, owner);
  const bool valid =
      refreshed && version(**output_backing) == output_version_before + 3u;
  std::fprintf(
      stderr,
      "DeviceVsm resident hash backend=%u valid=%u cold=%u warm=%u "
      "invalidate=%u refreshed=%u observations=%llu reuses=%llu queue=3\n",
      static_cast<unsigned>(backend), static_cast<unsigned>(valid),
      static_cast<unsigned>(cold), static_cast<unsigned>(warm),
      static_cast<unsigned>(invalidated), static_cast<unsigned>(refreshed),
      static_cast<unsigned long long>(
          owner == nullptr ? 0u : owner->output_hash_observation_count),
      static_cast<unsigned long long>(
          owner == nullptr ? 0u : owner->output_hash_reuse_count));
  return valid;
}

} // namespace rund_node_test_device_vsm_product::binary_test

#endif
