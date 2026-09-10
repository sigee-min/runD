#include "local.hpp"

#include "backing.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

constexpr std::size_t FrameElements = 16u;
constexpr std::size_t ElementCount = 73u;
constexpr std::size_t StageCount = 3u;

[[nodiscard]] bool contains(const std::span<const std::uint32_t> values,
                            const std::uint32_t value) noexcept {
  return std::find(values.begin(), values.end(), value) != values.end();
}

} // namespace

int CheckProductGraphLaterMultiHostWavefront(
    const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  DeviceVsmBypassScope route{*opened};
  if (!route || !route.device_vsm_available()) {
    return 2;
  }

  auto program =
      on(*opened)
          .input<std::uint64_t>(FrameElements)
          .zip_input<std::uint64_t>(FrameElements)
          .branch([](auto first, auto second) {
            const auto prefix =
                first.map("virtual-graph-later-multi-prefix",
                          [](auto value) { return value + std::uint64_t{1u}; });
            return zip(prefix, first, second)
                .map("virtual-graph-later-multi-join",
                     [](auto prior, auto first_value, auto second_value) {
                       return prior + first_value * std::uint64_t{2u} +
                              second_value * std::uint64_t{3u};
                     })
                .reduce(Reduce::Sum);
          })
          .compile();
  auto slices = program ? detail::graph_compile::compile_tiled_graph_slices(
                              detail::FlowAccess::state(*program))
                        : Result<detail::graph_compile::TiledGraphSlices>::fail(
                              Reason::PipelineInvalid);
  if (!program || !slices || slices->stages.size() != StageCount ||
      slices->input_resources.size() != 2u ||
      slices->stages[0u].inputs.size() != 1u ||
      slices->stages[1u].inputs.size() != 3u ||
      !contains(slices->stages[1u].inputs, slices->input_resources[0u]) ||
      !contains(slices->stages[1u].inputs, slices->input_resources[1u]) ||
      slices->stages[0u].outputs.size() != 1u ||
      !contains(slices->stages[1u].inputs,
                slices->stages[0u].outputs.front()) ||
      slices->stages[1u].outputs.size() != 1u ||
      slices->stages[2u].inputs != slices->stages[1u].outputs) {
    return 3;
  }

  std::array<std::uint64_t, ElementCount> first_values{};
  std::array<std::uint64_t, ElementCount> second_values{};
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    first_values[index] = index * 7u + 5u;
    second_values[index] = index * 11u + 3u;
  }
  auto first_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(first_values), FrameElements * sizeof(std::uint64_t));
  auto second_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(second_values), FrameElements * sizeof(std::uint64_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::uint64_t), sizeof(std::uint64_t));
  if (!first_backing->seed(std::as_bytes(std::span{first_values})) ||
      !second_backing->seed(std::as_bytes(std::span{second_values}))) {
    return 4;
  }
  auto first = virtual_buffer<std::uint64_t>(ElementCount, first_backing);
  auto second = virtual_buffer<std::uint64_t>(ElementCount, second_backing);
  auto output = virtual_buffer<std::uint64_t>(1u, output_backing);
  auto pipeline =
      first && second && output
          ? virtual_pipeline(*program, *first, *second, *output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint64_t(
                std::uint64_t, std::uint64_t)>>::fail(Reason::PipelineInvalid);
  if (!pipeline) {
    return 5;
  }

  const Stats before = pipeline->stats();
  const Status status = pipeline->run();
  const Stats after = pipeline->stats();
  const ResidencyStats residency = after.pipeline.residency;
  const std::uint64_t capacity = pipeline->plan().residency.frame_capacity;
  const BackingFacts first_facts = first_backing->facts();
  const BackingFacts second_facts = second_backing->facts();
  const BackingFacts output_facts = output_backing->facts();
  // This is an explicit required-route bypass oracle, not Host Graph
  // execution. A multi-input GraphReduction remains device-VSM-required;
  // disabling that route must stop before any execution or publication fact.
  const bool rejected =
      !status && status.reason() == Reason::BackendUnsupported &&
      capacity == 2u && residency.window_handoff_count == 0u &&
      residency.window_batch_count == 0u &&
      residency.window_queue_call_count == 0u && residency.epoch_count == 0u &&
      residency.page_in_count == 0u && residency.page_out_count == 0u &&
      residency.backing_read_bytes == 0u &&
      residency.backing_write_bytes == 0u && first_facts.read_count == 0u &&
      second_facts.read_count == 0u && output_facts.write_count == 0u &&
      output_facts.write_bytes == 0u &&
      after.command_submits == before.command_submits &&
      after.dispatches == before.dispatches &&
      after.publication == before.publication;
  if (!rejected) {
    std::fprintf(
        stderr,
        "virtual later required GraphReduction bypass backend=%u "
        "reason=%.*s k=%llu epochs=%llu page_in=%llu read=%llu "
        "first_reads=%llu/%llu second_reads=%llu/%llu page_out=%llu "
        "write=%llu submits=%llu dispatches=%llu publication=%llu/%llu\n",
        static_cast<unsigned>(backend), static_cast<int>(status.error().size()),
        status.error().data(), static_cast<unsigned long long>(capacity),
        static_cast<unsigned long long>(residency.epoch_count),
        static_cast<unsigned long long>(residency.page_in_count),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(first_facts.read_count),
        static_cast<unsigned long long>(first_facts.read_bytes),
        static_cast<unsigned long long>(second_facts.read_count),
        static_cast<unsigned long long>(second_facts.read_bytes),
        static_cast<unsigned long long>(residency.page_out_count),
        static_cast<unsigned long long>(residency.backing_write_bytes),
        static_cast<unsigned long long>(after.command_submits),
        static_cast<unsigned long long>(after.dispatches),
        static_cast<unsigned long long>(after.publication.commit_count),
        static_cast<unsigned long long>(after.publication.discard_count));
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
