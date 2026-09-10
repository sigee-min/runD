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

#include <cstdio>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::binary_test {
namespace {

using rund::compute::Backend;
using rund::compute::Reason;
using rund::compute::ResidencyConfig;
using rund::compute::Stats;
using rund::compute::Status;
using rund::compute::VirtualBacking;
using rund::compute::detail::VirtualBackingAccess;
using rund::compute::detail::VirtualPipelineAccess;

[[nodiscard]] std::uint64_t version(VirtualBacking &backing) noexcept {
  std::lock_guard lock{VirtualBackingAccess::gate(backing)};
  return VirtualBackingAccess::version(backing);
}

[[nodiscard]] std::uint64_t recovery(VirtualBacking &backing) noexcept {
  std::lock_guard lock{VirtualBackingAccess::gate(backing)};
  return VirtualBackingAccess::recovery_bytes(backing);
}

[[nodiscard]] bool run_resident(
    const Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, bool &unavailable) noexcept {
  using namespace rund::compute;
  unavailable = false;
  const std::uint64_t elements = pages * PageElements - 3u;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    unavailable = device.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program =
      on(*device)
          .map<std::uint32_t>("device-vsm-resident-product", PageElements,
                              [](auto value) { return value * 5u + 9u; })
          .compile();
  auto input_backing =
      resident_virtual_backing<std::uint32_t>(*device, elements);
  auto output_backing =
      resident_virtual_backing<std::uint32_t>(*device, elements);
  std::vector<std::uint32_t> input(static_cast<std::size_t>(elements));
  std::vector<std::uint32_t> expected(input.size());
  std::vector<std::uint32_t> observed(input.size());
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::uint32_t>(index * 7u + pages);
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
  if (!pipeline) {
    return false;
  }
  const auto state = VirtualPipelineAccess::state(*pipeline);
  std::uint64_t queue_before = 0u;
  if (state == nullptr || queue_counter == nullptr ||
      !queue_counter(state, queue_before)) {
    return false;
  }
  const std::uint64_t version_before = version(**output_backing);
  rund_node_test_device_vsm_product::RouteObservation observation{};
  const Status status =
      rund_node_test_device_vsm_product::RunThroughDeviceVsmProductRoute(
          state, observation);
  std::uint64_t queue_after = 0u;
  const auto owner = std::static_pointer_cast<
      detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const Stats stats = pipeline->stats();
  const auto &residency = stats.pipeline.residency;
  const std::uint64_t bytes = elements * sizeof(std::uint32_t);
  const bool output =
      status &&
      (*output_backing)
          ->read(0u, std::as_writable_bytes(std::span{observed})) &&
      observed == expected;
  const bool valid =
      output && queue_counter(state, queue_after) &&
      queue_after == queue_before + 1u && owner != nullptr &&
      owner->resident_inputs[0u] != nullptr &&
      owner->resident_output != nullptr && owner->input_staging[0u].empty() &&
      owner->output_staging.empty() &&
      owner->evidence->public_resident_input_count == 1u &&
      owner->evidence->whole_run_staged_input_count == 0u &&
      owner->evidence->public_resident_output &&
      !owner->evidence->whole_run_staged_output &&
      !owner->evidence->bounded_external_page_service &&
      ExactDeviceVsmEvidence(observation, pages, pages, bytes, bytes, 0u) &&
      stats.command_submits == 1u && stats.dispatches == 1u &&
      stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
      residency.backing_read_bytes == 0u &&
      residency.backing_write_bytes == 0u && residency.page_in_count == pages &&
      residency.page_out_count == pages && residency.page_in_bytes == bytes &&
      residency.page_out_bytes == bytes &&
      version(**output_backing) == version_before + 1u &&
      recovery(**output_backing) == 0u;
  std::fprintf(
      stderr,
      "DeviceVsm resident product backend=%u Q=%llu valid=%u output=%u "
      "queue=%llu/%llu submit=%llu dispatch=%llu host=%llu/%llu "
      "backing=%llu/%llu page=%llu/%llu final=%u route=%u/%u/%u/%s "
      "owner=%u resident=%u/%u staging=%llu/%llu\n",
      static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
      static_cast<unsigned>(valid), static_cast<unsigned>(output),
      static_cast<unsigned long long>(queue_before),
      static_cast<unsigned long long>(queue_after),
      static_cast<unsigned long long>(stats.command_submits),
      static_cast<unsigned long long>(stats.dispatches),
      static_cast<unsigned long long>(
          observation.evidence.native.host_service_turn_count),
      static_cast<unsigned long long>(
          observation.evidence.native.host_epoch_callback_count),
      static_cast<unsigned long long>(residency.backing_read_bytes),
      static_cast<unsigned long long>(residency.backing_write_bytes),
      static_cast<unsigned long long>(residency.page_in_count),
      static_cast<unsigned long long>(residency.page_out_count),
      static_cast<unsigned>(observation.evidence.final_received),
      static_cast<unsigned>(observation.production_route),
      static_cast<unsigned>(observation.prepared),
      static_cast<unsigned>(observation.executed),
      observation.prepare_reason == nullptr ? "null"
                                            : observation.prepare_reason,
      static_cast<unsigned>(owner != nullptr),
      static_cast<unsigned>(owner != nullptr &&
                            owner->resident_inputs[0u] != nullptr),
      static_cast<unsigned>(owner != nullptr &&
                            owner->resident_output != nullptr),
      static_cast<unsigned long long>(
          owner == nullptr ? 0u : owner->input_staging[0u].size()),
      static_cast<unsigned long long>(
          owner == nullptr ? 0u : owner->output_staging.size()));
  return valid;
}

} // namespace

bool RunResidentProductCases(
    const Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    bool unavailable = false;
    if (!run_resident(backend, queue_counter, pages, unavailable)) {
      return unavailable;
    }
    if (unavailable) {
      return true;
    }
  }
  bool unavailable = false;
  return RunResidentHashCacheCase(backend, queue_counter, unavailable) ||
         unavailable;
}

} // namespace rund_node_test_device_vsm_product::binary_test

#endif
