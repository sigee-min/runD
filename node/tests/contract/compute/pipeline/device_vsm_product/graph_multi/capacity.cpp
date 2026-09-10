#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"
#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::graph_multi_test {
namespace {

using Backing = rund_node_test_persistent_product::PersistentProductBacking;
using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

constexpr std::size_t InputCount = 6u;
constexpr std::uint64_t Pages = 5u;

struct PreparedMaximum final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::array<std::shared_ptr<Backing>, InputCount> inputs{};
  std::shared_ptr<Backing> output{};
  std::uint64_t expected{};
  std::uint64_t input_bytes{};
  rund::kernel::ReduceOp operation{rund::kernel::ReduceOp::Sum};
};

bool PrepareMaximum(const rund::compute::Backend backend,
                    const rund::kernel::ReduceOp operation,
                    PreparedMaximum &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  const std::uint64_t elements = Pages * PageElements - 3u;
  const std::size_t bytes =
      static_cast<std::size_t>(elements * sizeof(std::uint64_t));
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto mapped = on(*opened)
                    .input<std::uint64_t>(PageElements)
                    .zip_input<std::uint64_t>(PageElements)
                    .zip_input<std::uint64_t>(PageElements)
                    .zip_input<std::uint64_t>(PageElements)
                    .zip_input<std::uint64_t>(PageElements)
                    .zip_input<std::uint64_t>(PageElements)
                    .map("device-vsm-graph-six-input-map",
                         [](auto a, auto b, auto c, auto d, auto e, auto f) {
                           return a + b + c + d + e + f;
                         });
  auto program = [&]() {
    if (operation == rund::kernel::ReduceOp::CountNonzero) {
      return std::move(mapped).count().compile();
    }
    if (operation == rund::kernel::ReduceOp::Min) {
      return std::move(mapped).reduce(Reduce::Min).compile();
    }
    if (operation == rund::kernel::ReduceOp::Max) {
      return std::move(mapped).reduce(Reduce::Max).compile();
    }
    return std::move(mapped).reduce(Reduce::Sum).compile();
  }();
  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> mapped_values(static_cast<std::size_t>(elements));
  for (std::size_t input = 0u; input < InputCount; ++input) {
    prepared.inputs[input] = std::make_shared<Backing>(bytes);
    values[input].resize(static_cast<std::size_t>(elements));
    for (std::size_t index = 0u; index < values[input].size(); ++index) {
      values[input][index] =
          1u + input * 13u + index * static_cast<std::uint64_t>(input + 3u);
      if (mapped_values[index] >
          std::numeric_limits<std::uint64_t>::max() - values[input][index]) {
        return false;
      }
      mapped_values[index] += values[input][index];
    }
    if (prepared.inputs[input] == nullptr ||
        !prepared.inputs[input]->seed(
            std::as_bytes(std::span{values[input]}))) {
      return false;
    }
  }
  prepared.operation = operation;
  prepared.expected = operation == rund::kernel::ReduceOp::Min
                          ? std::numeric_limits<std::uint64_t>::max()
                          : 0u;
  for (const std::uint64_t value : mapped_values) {
    if (operation == rund::kernel::ReduceOp::Sum) {
      if (prepared.expected >
          std::numeric_limits<std::uint64_t>::max() - value) {
        return false;
      }
      prepared.expected += value;
    } else if (operation == rund::kernel::ReduceOp::CountNonzero) {
      prepared.expected += static_cast<std::uint64_t>(value != 0u);
    } else if (operation == rund::kernel::ReduceOp::Min) {
      prepared.expected = std::min(prepared.expected, value);
    } else {
      prepared.expected = std::max(prepared.expected, value);
    }
  }
  prepared.output = std::make_shared<Backing>(sizeof(std::uint64_t));
  prepared.input_bytes = elements * sizeof(std::uint64_t);
  if (!program || prepared.output == nullptr) {
    return false;
  }
  auto a =
      detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.inputs[0u]);
  auto b =
      detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.inputs[1u]);
  auto c =
      detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.inputs[2u]);
  auto d =
      detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.inputs[3u]);
  auto e =
      detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.inputs[4u]);
  auto f =
      detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                  detail::Type::U64, {}, prepared.inputs[5u]);
  auto output = detail::make_virtual_buffer(
      1u, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.output);
  const auto program_state = detail::ProgramAccess::state(*program);
  if (!a || !b || !c || !d || !e || !f || !output || program_state == nullptr) {
    return false;
  }
  const std::array buffers{std::move(a).value(), std::move(b).value(),
                           std::move(c).value(), std::move(d).value(),
                           std::move(e).value(), std::move(f).value()};
  auto state = detail::prepare_virtual_pipeline(
      program_state, buffers, std::move(output).value(),
      ResidencyConfig{.device_resident_bytes = 1u << 20u,
                      .host_resident_bytes = 1u << 20u});
  if (!state || state.value() == nullptr) {
    std::fprintf(stderr,
                 "DeviceVsm Graph six-input prepare backend=%u reason=%u "
                 "native=%s\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(state.reason()),
                 state.location().native_reason_key == nullptr
                     ? "null"
                     : state.location().native_reason_key);
    return false;
  }
  prepared.state = std::move(state).value();
  const auto &plan = prepared.state->pipeline->residency->tiled_graph();
  return prepared.state->geometry.route ==
             detail::VirtualRoute::GraphReduction &&
         prepared.state->geometry.device_vsm_required &&
         prepared.state->input_count == InputCount &&
         prepared.state->graph_input_resource_count == InputCount &&
         prepared.state->device_vsm_semantic_pipeline == nullptr &&
         plan.stages().size() == 2u && plan.resources().size() == 8u &&
         plan.stages().front().ports.size() == InputCount + 1u;
}

bool ExactOutput(const PreparedMaximum &prepared) noexcept {
  std::uint64_t observed = 0u;
  return prepared.output != nullptr &&
         prepared.output->observe(
             std::as_writable_bytes(std::span<std::uint64_t>{&observed, 1u})) &&
         observed == prepared.expected;
}

bool RunMaximumCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const rund::kernel::ReduceOp operation) noexcept {
  using namespace rund_node_test_persistent_product;
  PreparedMaximum prepared{};
  bool unavailable = false;
  if (!PrepareMaximum(backend, operation, prepared, unavailable)) {
    return unavailable;
  }
  const PublicationSnapshot before_map =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot before_reduce =
      SnapshotPublication(rund::compute::detail::graph_terminal_pipeline(
          *prepared.state, 0u));
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
  const PublicationSnapshot after_map =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_reduce =
      SnapshotPublication(rund::compute::detail::graph_terminal_pipeline(
          *prepared.state, 0u));
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::GraphMapReduce &&
      owner->proof->graph_map_reduce.semantic.op == operation &&
      owner->proof->plan.input_buffer_count == InputCount &&
      owner->proof->plan.output_buffer_count == 1u &&
      owner->proof->residents.input_count == InputCount &&
      owner->proof->residents.output_count == 1u &&
      owner->proof->residents.count == InputCount + 1u &&
      ExactDeviceVsmEvidence(observation, Pages, 1u,
                             InputCount * prepared.input_bytes,
                             sizeof(std::uint64_t), 0u) &&
      ExactPublication(before_map, after_map, before_reduce, after_reduce,
                       version_before, BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u &&
      stats.page_in_count == InputCount * Pages && stats.page_out_count == 1u &&
      stats.backing_read_bytes == InputCount * prepared.input_bytes &&
      stats.backing_write_bytes == sizeof(std::uint64_t);
  std::fprintf(
      stderr,
      "DeviceVsm Graph six-input backend=%u op=%u valid=%u submit=%llu "
      "epoch_submit=%llu host_turn=%llu host_callback=%llu generated=%llu "
      "completed=%llu final=%llu inputs=%u pages=%llu read=%llu "
      "publish=%llu/%llu/%llu\n",
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
      static_cast<unsigned long long>(
          observation.evidence.native.generated_epochs),
      static_cast<unsigned long long>(
          observation.evidence.native.completed_epochs),
      static_cast<unsigned long long>(
          observation.evidence.native.final_callback_count),
      owner == nullptr || owner->proof == nullptr
          ? 0u
          : owner->proof->residents.input_count,
      static_cast<unsigned long long>(stats.page_in_count),
      static_cast<unsigned long long>(stats.backing_read_bytes),
      static_cast<unsigned long long>(
          observation.evidence.authority_accept_count),
      static_cast<unsigned long long>(
          observation.evidence.pipeline_terminal_count),
      static_cast<unsigned long long>(
          observation.evidence.backing_publication_count));
  return valid;
}

} // namespace

bool RunGraphMaximumInputProduct(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  for (const rund::kernel::ReduceOp operation :
       {rund::kernel::ReduceOp::Sum, rund::kernel::ReduceOp::CountNonzero,
        rund::kernel::ReduceOp::Min, rund::kernel::ReduceOp::Max}) {
    if (!RunMaximumCase(backend, queue_counter, operation)) {
      return false;
    }
  }
  return true;
}

} // namespace rund_node_test_device_vsm_product::graph_multi_test

#endif
