#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::graph_multi_test {

bool PrepareGraphMulti(const rund::compute::Backend backend,
                       const std::uint64_t pages,
                       const rund::kernel::ReduceOp operation,
                       PreparedGraphMulti &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  if (pages < 2u) {
    return false;
  }
  const std::uint64_t elements = pages * PageElements - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto mapped =
      on(*opened)
          .input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .map("device-vsm-graph-multi-map", [](auto first, auto second) {
            return (first ^ 0x55ull) + second * 3ull;
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
  prepared.first = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.second = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      sizeof(std::uint64_t));
  std::vector<std::uint64_t> first(static_cast<std::size_t>(elements));
  std::vector<std::uint64_t> second(first.size());
  prepared.operation = operation;
  prepared.expected = operation == rund::kernel::ReduceOp::Min
                          ? std::numeric_limits<std::uint64_t>::max()
                          : 0u;
  for (std::size_t index = 0u; index < first.size(); ++index) {
    first[index] = index * 7u + pages;
    second[index] = index * 11u + 3u;
    const std::uint64_t mapped_value =
        (first[index] ^ 0x55ull) + second[index] * 3ull;
    if (operation == rund::kernel::ReduceOp::Sum) {
      prepared.expected += mapped_value;
    } else if (operation == rund::kernel::ReduceOp::CountNonzero) {
      prepared.expected += static_cast<std::uint64_t>(mapped_value != 0u);
    } else if (operation == rund::kernel::ReduceOp::Min) {
      prepared.expected = std::min(prepared.expected, mapped_value);
    } else {
      prepared.expected = std::max(prepared.expected, mapped_value);
    }
  }
  prepared.input_bytes = elements * sizeof(std::uint64_t);
  if (!program || prepared.first == nullptr || prepared.second == nullptr ||
      prepared.output == nullptr ||
      !prepared.first->seed(std::as_bytes(std::span{first})) ||
      !prepared.second->seed(std::as_bytes(std::span{second}))) {
    return false;
  }
  auto first_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.first);
  auto second_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.second);
  auto output_buffer = detail::make_virtual_buffer(
      1u, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.output);
  const auto program_state = detail::ProgramAccess::state(*program);
  if (!first_buffer || !second_buffer || !output_buffer ||
      program_state == nullptr) {
    return false;
  }
  const std::array inputs{std::move(first_buffer).value(),
                          std::move(second_buffer).value()};
  auto state = detail::prepare_virtual_pipeline(
      program_state, inputs, std::move(output_buffer).value(),
      ResidencyConfig{.device_resident_bytes = 1u << 20u,
                      .host_resident_bytes = 1u << 20u});
  if (!state || state.value() == nullptr) {
    std::fprintf(stderr,
                 "DeviceVsm Graph multi prepare backend=%u Q=%llu reason=%u "
                 "location=%u/%u/%u native=%s\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(pages),
                 static_cast<unsigned>(state.reason()), state.location().step,
                 state.location().iteration, state.location().node,
                 state.location().native_reason_key == nullptr
                     ? "null"
                     : state.location().native_reason_key);
    return false;
  }
  prepared.state = std::move(state).value();
  const auto &plan = prepared.state->pipeline->residency->tiled_graph();
  const bool valid =
      prepared.state->geometry.route == detail::VirtualRoute::GraphReduction &&
      prepared.state->geometry.device_vsm_required &&
      prepared.state->input_count == 2u &&
      prepared.state->graph_input_resource_count == 2u &&
      prepared.state->device_vsm_semantic_pipeline == nullptr &&
      plan.stages().size() == 2u && plan.resources().size() == 4u &&
      plan.stages().front().ports.size() == 3u;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm Graph multi shape backend=%u Q=%llu route=%u "
        "required=%u inputs=%zu graph_inputs=%zu stages=%zu "
        "resources=%zu ports=%zu semantic=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned>(prepared.state->geometry.route),
        static_cast<unsigned>(prepared.state->geometry.device_vsm_required),
        prepared.state->input_count, prepared.state->graph_input_resource_count,
        plan.stages().size(), plan.resources().size(),
        plan.stages().empty() ? 0u : plan.stages().front().ports.size(),
        static_cast<unsigned>(prepared.state->device_vsm_semantic_pipeline !=
                              nullptr));
  }
  return valid;
}

bool ExactGraphMultiOutput(const PreparedGraphMulti &prepared) noexcept {
  std::uint64_t observed = 0u;
  return prepared.output != nullptr &&
         prepared.output->observe(
             std::as_writable_bytes(std::span<std::uint64_t>{&observed, 1u})) &&
         observed == prepared.expected;
}

} // namespace rund_node_test_device_vsm_product::graph_multi_test

#endif
