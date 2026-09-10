#include "internal.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>
#include <span>

namespace rund_node_test_virtual::product::graph_pointwise_wide_host {
namespace {

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] std::shared_ptr<Owner> owner(const Case &test_case) noexcept {
  return test_case.state == nullptr
             ? std::shared_ptr<Owner>{}
             : std::static_pointer_cast<Owner>(
                   test_case.state->device_vsm_product_cache);
}

[[nodiscard]] bool proof_valid(const Owner &value) noexcept {
  using namespace rund::node::accel::detail;
  const auto &proof = value.proof;
  if (proof == nullptr ||
      proof->topology != DeviceVsmTopology::GraphPointwise ||
      proof->plan.input_buffer_count != InputCount ||
      proof->plan.output_buffer_count != 1u ||
      proof->residents.input_count != InputCount ||
      proof->residents.output_count != 1u ||
      proof->residents.count != InputCount + 1u ||
      proof->graph_pointwise.stage_count != StageCount ||
      proof->graph_pointwise.topology.external_input_count != InputCount ||
      proof->graph_pointwise.topology.stages[0u].input_count != InputCount ||
      proof->graph_pointwise.topology.stages[1u].input_count != 1u) {
    return false;
  }
  for (std::size_t input = 0u; input < InputCount; ++input) {
    const DeviceVsmGraphValueSource source =
        proof->graph_pointwise.topology.stages[0u].inputs[input];
    if (source.kind != DeviceVsmGraphValueSourceKind::ExternalInput ||
        source.index != input) {
      return false;
    }
  }
  const DeviceVsmGraphValueSource terminal =
      proof->graph_pointwise.topology.stages[1u].inputs[0u];
  return terminal.kind == DeviceVsmGraphValueSourceKind::StageOutput &&
         terminal.index == 0u;
}

[[nodiscard]] bool native_valid(const Owner &value) noexcept {
  const auto &evidence = value.evidence;
  return evidence != nullptr && evidence->public_handoff_count == 1u &&
         evidence->authority_accept_count == 1u &&
         evidence->pipeline_terminal_count == StageCount &&
         evidence->backing_publication_count == 1u &&
         evidence->final_received && !evidence->quarantined &&
         evidence->native.page_count == PageCount &&
         evidence->native.generated_epochs == PageCount &&
         evidence->native.completed_epochs == PageCount &&
         evidence->native.native_submit_count == 1u &&
         evidence->native.epoch_native_submit_count == 0u &&
         evidence->native.payload_dispatch_count == 1u &&
         evidence->native.host_service_turn_count == 0u &&
         evidence->native.host_epoch_callback_count == 0u &&
         evidence->native.final_callback_count == 1u;
}

[[nodiscard]] bool backing_valid(const Case &test_case,
                                 const std::uint64_t initial_version) noexcept {
  using rund::compute::detail::VirtualBackingAccess;
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  constexpr std::size_t LogicalBytes = ElementCount * sizeof(std::uint64_t);
  for (const auto &backing : test_case.input_backings) {
    const BackingFacts facts = backing->facts();
    if (facts.read_count != PageCount || facts.read_bytes != LogicalBytes) {
      return false;
    }
  }
  const BackingFacts output = test_case.output_backing->facts();
  return output.write_count == PageCount &&
         output.write_bytes == LogicalBytes &&
         VirtualBackingAccess::version(*test_case.output_backing) ==
             initial_version + 1u &&
         VirtualBackingAccess::recovery_bytes(*test_case.output_backing) == 0u;
}

[[nodiscard]] bool output_valid(Case &test_case) noexcept {
  std::vector<std::uint64_t> observed(test_case.expected.size());
  return test_case.output_backing->observe(
             std::as_writable_bytes(std::span{observed})) &&
         std::equal(observed.begin(), observed.end(),
                    test_case.expected.begin(), test_case.expected.end()) &&
         test_case.output_backing->tail_poisoned();
}

} // namespace

std::array<std::uint64_t, StageCount>
device_stage_generations(const Case &test_case) noexcept {
  using rund::compute::detail::residency::Pool;
  std::array<std::uint64_t, StageCount> result{};
  if (test_case.state == nullptr ||
      test_case.state->graph_pipelines.size() != StageCount * Pool::BankCount) {
    result.fill(std::numeric_limits<std::uint64_t>::max());
    return result;
  }
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    const auto &pipeline =
        test_case.state->graph_pipelines[stage * Pool::BankCount];
    if (pipeline == nullptr || pipeline->publication == nullptr) {
      result.fill(std::numeric_limits<std::uint64_t>::max());
      return result;
    }
    std::lock_guard state_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    result[stage] = pipeline->publication->generation;
  }
  return result;
}

bool validate_device_case(
    Case &test_case, const rund::compute::Backend backend,
    const rund::compute::Status &status, const std::uint64_t initial_version,
    const std::array<std::uint64_t, StageCount> &initial_generations) noexcept {
  using namespace rund::compute;
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  constexpr std::size_t LogicalBytes = ElementCount * sizeof(std::uint64_t);
  const Stats stats = test_case.pipeline.stats();
  const ResidencyStats residency = stats.pipeline.residency;
  const std::shared_ptr<Owner> retained = owner(test_case);
  const auto final_generations = device_stage_generations(test_case);
  bool generations_valid = true;
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    generations_valid =
        generations_valid &&
        initial_generations[stage] !=
            std::numeric_limits<std::uint64_t>::max() &&
        initial_generations[stage] !=
            std::numeric_limits<std::uint64_t>::max() - 1u &&
        final_generations[stage] == initial_generations[stage] + 1u;
  }
  const bool valid =
      backend != Backend::Cpu && status && retained != nullptr &&
      proof_valid(*retained) && native_valid(*retained) &&
      backing_valid(test_case, initial_version) && generations_valid &&
      test_case.pipeline.plan().residency.frame_capacity == 2u &&
      residency.window_handoff_count == 1u &&
      residency.window_batch_count == 1u &&
      residency.window_queue_call_count == 1u &&
      residency.page_in_count == InputCount * PageCount &&
      residency.backing_read_bytes == InputCount * LogicalBytes &&
      residency.page_out_count == PageCount &&
      residency.backing_write_bytes == LogicalBytes &&
      stats.command_submits == 1u && stats.dispatches == 1u &&
      stats.final_dispatches == 1u && output_valid(test_case);
  std::fprintf(
      stderr,
      "Graph wide GPU backend=%u valid=%u owner=%u submit=%llu dispatch=%llu "
      "page=%llu/%llu read=%llu final=%llu publish=%llu/%llu/%llu\n",
      static_cast<unsigned>(backend), static_cast<unsigned>(valid),
      static_cast<unsigned>(retained != nullptr),
      static_cast<unsigned long long>(stats.command_submits),
      static_cast<unsigned long long>(stats.dispatches),
      static_cast<unsigned long long>(residency.page_in_count),
      static_cast<unsigned long long>(residency.page_out_count),
      static_cast<unsigned long long>(residency.backing_read_bytes),
      retained == nullptr || retained->evidence == nullptr
          ? 0u
          : static_cast<unsigned long long>(
                retained->evidence->native.final_callback_count),
      retained == nullptr || retained->evidence == nullptr
          ? 0u
          : static_cast<unsigned long long>(
                retained->evidence->authority_accept_count),
      retained == nullptr || retained->evidence == nullptr
          ? 0u
          : static_cast<unsigned long long>(
                retained->evidence->pipeline_terminal_count),
      retained == nullptr || retained->evidence == nullptr
          ? 0u
          : static_cast<unsigned long long>(
                retained->evidence->backing_publication_count));
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_wide_host
