#include "local.hpp"

#include "backing.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

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
constexpr std::size_t MultiPageCount =
    (ElementCount + FrameElements - 1u) / FrameElements;

} // namespace

int CheckProductGraphMultiHostWavefront(const rund::compute::Backend backend) {
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
  auto program = on(*opened)
                     .input<std::uint64_t>(FrameElements)
                     .zip_input<std::uint64_t>(FrameElements)
                     .map("virtual-graph-multi-host-map",
                          [](auto first, auto second) {
                            return (first ^ std::uint64_t{0x55u}) +
                                   second * std::uint64_t{3u};
                          })
                     .reduce(Reduce::Sum)
                     .compile();
  std::array<std::uint64_t, ElementCount> first{};
  std::array<std::uint64_t, ElementCount> second{};
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    first[index] = index * 7u + 5u;
    second[index] = index * 11u + 3u;
  }
  auto first_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(first), FrameElements * sizeof(std::uint64_t));
  auto second_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(second), FrameElements * sizeof(std::uint64_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::uint64_t), sizeof(std::uint64_t));
  if (!program || !first_backing->seed(std::as_bytes(std::span{first})) ||
      !second_backing->seed(std::as_bytes(std::span{second}))) {
    return 3;
  }
  auto first_buffer =
      virtual_buffer<std::uint64_t>(ElementCount, first_backing);
  auto second_buffer =
      virtual_buffer<std::uint64_t>(ElementCount, second_backing);
  auto output_buffer = virtual_buffer<std::uint64_t>(1u, output_backing);
  auto pipeline =
      first_buffer && second_buffer && output_buffer
          ? virtual_pipeline(*program, *first_buffer, *second_buffer,
                             *output_buffer, ResidencyConfig{})
          : Result<VirtualPipeline<std::uint64_t(
                std::uint64_t, std::uint64_t)>>::fail(Reason::PipelineInvalid);
  const Stats before = pipeline ? pipeline->stats() : Stats{};
  const Status status =
      pipeline ? pipeline->run() : Status::fail(pipeline.reason());
  const Stats after = pipeline ? pipeline->stats() : Stats{};
  const ResidencyStats residency =
      pipeline ? pipeline->stats().pipeline.residency : ResidencyStats{};
  const std::uint64_t capacity =
      pipeline ? pipeline->plan().residency.frame_capacity : 0u;
  const std::uint64_t batches =
      capacity == 0u ? 0u : (MultiPageCount + capacity - 1u) / capacity;
  const BackingFacts first_facts = first_backing->facts();
  const BackingFacts second_facts = second_backing->facts();
  const BackingFacts output_facts = output_backing->facts();
  // This is an explicit required-route bypass oracle, not Host Graph
  // execution. A multi-input GraphReduction remains device-VSM-required;
  // disabling that route must stop before any execution or publication fact.
  const bool rejected =
      pipeline && !status && status.reason() == Reason::BackendUnsupported &&
      capacity == 2u && batches == (MultiPageCount + 1u) / 2u &&
      residency.window_handoff_count == 0u &&
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
        "virtual required GraphReduction bypass backend=%u reason=%.*s "
        "k=%llu epochs=%llu page_in=%llu read=%llu first=%llu/%llu "
        "second=%llu/%llu page_out=%llu write=%llu submits=%llu "
        "dispatches=%llu publication=%llu/%llu\n",
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
    return 4;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
