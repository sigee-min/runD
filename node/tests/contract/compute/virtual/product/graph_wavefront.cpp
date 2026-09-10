#include "local.hpp"

#include "backing.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

constexpr std::size_t FrameElements = 16u;
constexpr std::size_t ElementCount = 73u;
constexpr std::size_t GraphPageCount =
    (ElementCount + FrameElements - 1u) / FrameElements;
constexpr std::size_t PhysicalStageCount = 4u;

[[nodiscard]] bool observe_scalar(MemoryVirtualBacking &backing,
                                  std::uint64_t &value) noexcept {
  std::array<std::byte, sizeof(value)> bytes{};
  if (!backing.observe(bytes)) {
    return false;
  }
  std::memcpy(&value, bytes.data(), bytes.size());
  return true;
}

} // namespace

int CheckProductGraphWavefront(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return device.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  auto program =
      on(*device)
          .input<std::uint64_t>(FrameElements)
          .branch([](auto values) {
            const auto left =
                values.map("virtual-graph-wavefront-left", [](auto value) {
                  return value + std::uint64_t{1u};
                });
            const auto right =
                values.map("virtual-graph-wavefront-right", [](auto value) {
                  return value * std::uint64_t{2u};
                });
            return zip(left, right)
                .map("virtual-graph-wavefront-join",
                     [](auto first, auto second) { return first + second; })
                .reduce(Reduce::Sum);
          })
          .compile();
  auto slices = program ? detail::graph_compile::compile_tiled_graph_slices(
                              detail::FlowAccess::state(*program))
                        : Result<detail::graph_compile::TiledGraphSlices>::fail(
                              Reason::PipelineInvalid);
  if (!program || !slices || slices->stages.size() != PhysicalStageCount ||
      slices->resources.size() != 5u ||
      slices->stages[0u].inputs != slices->stages[1u].inputs ||
      slices->stages[2u].inputs.size() != 2u ||
      slices->stages[3u].inputs != slices->stages[2u].outputs) {
    return 2;
  }

  std::array<std::uint64_t, ElementCount> values{};
  std::uint64_t expected = 0u;
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = (index * 17u + 5u) % 103u;
    expected += values[index] * 3u + 1u;
  }
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(values), FrameElements * sizeof(std::uint64_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::uint64_t), sizeof(std::uint64_t));
  if (!input_backing->seed(std::as_bytes(std::span{values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::uint64_t>(ElementCount, input_backing);
  auto output = virtual_buffer<std::uint64_t>(1u, output_backing);
  auto pipeline =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                Reason::PipelineInvalid);
  const Status status =
      pipeline ? pipeline->run() : Status::fail(pipeline.reason());
  std::uint64_t observed = 0u;
  const Stats stats = pipeline ? pipeline->stats() : Stats{};
  const ResidencyStats residency = stats.pipeline.residency;
  const std::uint64_t frame_capacity =
      pipeline ? pipeline->plan().residency.frame_capacity : 0u;
  const std::uint64_t batches =
      frame_capacity == 0u
          ? 0u
          : (GraphPageCount + frame_capacity - 1u) / frame_capacity;
  const bool device_vsm = backend != Backend::Cpu &&
                          residency.window_handoff_count == 1u &&
                          residency.window_batch_count == 1u &&
                          residency.window_queue_call_count == 1u;
  const std::uint64_t expected_epochs =
      device_vsm ? batches : batches * PhysicalStageCount;
  if (!pipeline || !status || frame_capacity != 2u ||
      (backend != Backend::Cpu && !device_vsm) ||
      !observe_scalar(*output_backing, observed) || observed != expected ||
      residency.epoch_count != expected_epochs ||
      residency.page_in_count != GraphPageCount ||
      residency.backing_read_bytes != sizeof(values) ||
      residency.page_out_count != 1u ||
      residency.backing_write_bytes != sizeof(std::uint64_t) ||
      (device_vsm && (stats.command_submits != 1u || stats.dispatches != 1u ||
                      stats.final_dispatches != 1u))) {
    std::fprintf(
        stderr,
        "virtual Graph wavefront backend=%u reason=%.*s observed=%llu "
        "expected=%llu stages=%zu k=%llu epochs=%llu expected_epochs=%llu "
        "hits=%llu page_in=%llu backing_read=%llu page_out=%llu "
        "backing_write=%llu\n",
        static_cast<unsigned>(backend), static_cast<int>(status.error().size()),
        status.error().data(), static_cast<unsigned long long>(observed),
        static_cast<unsigned long long>(expected), slices->stages.size(),
        static_cast<unsigned long long>(frame_capacity),
        static_cast<unsigned long long>(residency.epoch_count),
        static_cast<unsigned long long>(expected_epochs),
        static_cast<unsigned long long>(residency.cache_hit_count),
        static_cast<unsigned long long>(residency.page_in_count),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(residency.page_out_count),
        static_cast<unsigned long long>(residency.backing_write_bytes));
    return 4;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
