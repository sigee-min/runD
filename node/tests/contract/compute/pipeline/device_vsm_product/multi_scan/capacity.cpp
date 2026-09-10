#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"
#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::multi_scan_test {
namespace {

using Backing = rund_node_test_persistent_product::PersistentProductBacking;
using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

constexpr std::size_t InputCount = 7u;
constexpr std::uint64_t Pages = 5u;

struct MaximumInputScan final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::array<std::shared_ptr<Backing>, InputCount> inputs{};
  std::shared_ptr<Backing> output{};
  std::vector<std::uint64_t> expected{};
  std::uint64_t elements{};
};

[[nodiscard]] bool prepare(const rund::compute::Backend backend,
                           const rund::kernel::ScanOp operation,
                           MaximumInputScan &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  const bool exclusive = operation == rund::kernel::ScanOp::ExclusiveSum;
  const std::uint64_t payload = PageElements - (exclusive ? 1u : 0u);
  prepared.elements = Pages * payload - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program = on(*opened)
                     .input<std::uint64_t>(PageElements)
                     .zip_input<std::uint64_t>(PageElements)
                     .zip_input<std::uint64_t>(PageElements)
                     .zip_input<std::uint64_t>(PageElements)
                     .zip_input<std::uint64_t>(PageElements)
                     .zip_input<std::uint64_t>(PageElements)
                     .zip_input<std::uint64_t>(PageElements)
                     .map("device-vsm-seven-input-map-scan",
                          [](auto a, auto b, auto c, auto d, auto e, auto f,
                             auto g) { return a + b + c + d + e + f + g; })
                     .scan(exclusive ? Scan::ExclusiveSum : Scan::InclusiveSum)
                     .compile();
  if (!program) {
    std::fprintf(
        stderr, "DeviceVsm maximum Scan compile backend=%u op=%u reason=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(operation),
        static_cast<unsigned>(program.reason()));
    return false;
  }
  const std::size_t elements = static_cast<std::size_t>(prepared.elements);
  const std::size_t bytes = elements * sizeof(std::uint64_t);
  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> mapped(elements, 0u);
  std::array<std::shared_ptr<detail::VirtualBufferState>, InputCount> inputs{};
  for (std::size_t input = 0u; input < InputCount; ++input) {
    prepared.inputs[input] = std::make_shared<Backing>(bytes);
    values[input].resize(elements);
    for (std::size_t index = 0u; index < elements; ++index) {
      values[input][index] =
          1u + input * 13u + index * static_cast<std::uint64_t>(input + 3u);
      mapped[index] += values[input][index];
    }
    if (prepared.inputs[input] == nullptr ||
        !prepared.inputs[input]->seed(
            std::as_bytes(std::span{values[input]}))) {
      return false;
    }
    auto buffer = detail::make_virtual_buffer(
        prepared.elements, sizeof(std::uint64_t), detail::Type::U64, {},
        prepared.inputs[input]);
    if (!buffer) {
      return false;
    }
    inputs[input] = std::move(buffer).value();
  }
  prepared.expected.resize(elements);
  std::uint64_t prefix = 0u;
  for (std::size_t index = 0u; index < elements; ++index) {
    if (!exclusive) {
      prefix += mapped[index];
    }
    prepared.expected[index] = prefix;
    if (exclusive) {
      prefix += mapped[index];
    }
  }
  prepared.output = std::make_shared<Backing>(bytes);
  auto output =
      detail::make_virtual_buffer(prepared.elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.output);
  const std::shared_ptr<detail::ProgramState> program_state =
      detail::ProgramAccess::state(*program);
  if (prepared.output == nullptr || !output || program_state == nullptr) {
    return false;
  }
  auto state = detail::prepare_virtual_pipeline(
      program_state, inputs, std::move(output).value(), ResidencyConfig{});
  if (!state || state.value() == nullptr) {
    std::fprintf(
        stderr, "DeviceVsm maximum Scan prepare backend=%u op=%u reason=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(operation),
        static_cast<unsigned>(state.reason()));
    return false;
  }
  prepared.state = std::move(state).value();
  const bool valid =
      prepared.state->geometry.route == detail::VirtualRoute::Scan &&
      prepared.state->geometry.device_vsm_required &&
      prepared.state->input_count == InputCount &&
      prepared.state->pipeline != nullptr &&
      prepared.state->alternate_pipeline != nullptr &&
      prepared.state->pipeline->residency == nullptr &&
      prepared.state->alternate_pipeline->residency == nullptr;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm maximum Scan shape backend=%u op=%u route=%u "
        "required=%u inputs=%llu pipeline=%u alternate=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(operation),
        static_cast<unsigned>(prepared.state->geometry.route),
        static_cast<unsigned>(prepared.state->geometry.device_vsm_required),
        static_cast<unsigned long long>(prepared.state->input_count),
        static_cast<unsigned>(prepared.state->pipeline != nullptr),
        static_cast<unsigned>(prepared.state->alternate_pipeline != nullptr));
  }
  return valid;
}

[[nodiscard]] bool exact_output(const MaximumInputScan &prepared) {
  std::vector<std::byte> observed(prepared.expected.size() *
                                  sizeof(std::uint64_t));
  return prepared.output != nullptr && prepared.output->observe(observed) &&
         std::memcmp(observed.data(), prepared.expected.data(),
                     observed.size()) == 0;
}

} // namespace

bool RunMaximumInputScan(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const rund::kernel::ScanOp operation, bool &unavailable) noexcept {
  MaximumInputScan prepared{};
  if (!prepare(backend, operation, prepared, unavailable)) {
    return unavailable;
  }
  const PublicationSnapshot before_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot before_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (queue_counter == nullptr ||
      !queue_counter(prepared.state, queue_before)) {
    return false;
  }
  RouteObservation observation{};
  const rund::compute::Status status =
      RunThroughDeviceVsmProductRoute(prepared.state, observation);
  std::uint64_t queue_after = 0u;
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const std::uint64_t bytes = prepared.elements * sizeof(std::uint64_t);
  const std::uint64_t overlap = operation == rund::kernel::ScanOp::ExclusiveSum
                                    ? (Pages - 1u) * sizeof(std::uint64_t)
                                    : 0u;
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && exact_output(prepared) &&
      owner != nullptr && owner->proof != nullptr &&
      owner->input_count == InputCount &&
      owner->proof->residents.input_count == InputCount &&
      owner->proof->residents.output_count == 1u &&
      owner->proof->plan.input_buffer_count == InputCount &&
      owner->proof->plan.output_buffer_count == 1u &&
      owner->proof->scan.stage_count == 2u &&
      ExactDeviceVsmEvidence(observation, Pages, Pages, InputCount * bytes,
                             bytes, overlap) &&
      ExactPublication(before_primary,
                       SnapshotPublication(prepared.state->pipeline),
                       before_alternate,
                       SnapshotPublication(prepared.state->alternate_pipeline),
                       version_before, BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.page_in_count == InputCount * Pages &&
      stats.page_out_count == Pages &&
      stats.backing_read_bytes == InputCount * bytes &&
      stats.backing_write_bytes == bytes;
  std::fprintf(
      stderr,
      "DeviceVsm maximum Scan backend=%u op=%u valid=%u submit=%llu "
      "epoch_submit=%llu host_turn=%llu host_callback=%llu inputs=%llu "
      "pages=%llu read=%llu final=%u\n",
      static_cast<unsigned>(backend), static_cast<unsigned>(operation),
      static_cast<unsigned>(valid),
      static_cast<unsigned long long>(
          observation.evidence.native.native_submit_count),
      static_cast<unsigned long long>(
          observation.evidence.native.epoch_native_submit_count),
      static_cast<unsigned long long>(
          observation.evidence.native.host_service_turn_count),
      static_cast<unsigned long long>(
          observation.evidence.native.host_epoch_callback_count),
      static_cast<unsigned long long>(InputCount),
      static_cast<unsigned long long>(stats.page_in_count),
      static_cast<unsigned long long>(stats.backing_read_bytes),
      static_cast<unsigned>(observation.evidence.final_received));
  return valid;
}

} // namespace rund_node_test_device_vsm_product::multi_scan_test

#endif
