#include "window/local.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {

using namespace window_detail;

int CheckProductWindow(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto program =
      on(*device)
          .map<std::int32_t>("virtual-product-window-pre", FrameElements,
                             [](auto input) { return input + 3; })
          .window(WindowSpec{
              .op = Window::Sum, .radius = Radius, .edge = WindowEdge::Clamp})
          .map("virtual-product-window-post",
               [](auto input) { return input * 2; })
          .compile();
  if (!program || program->graph().nodes.size() != 3u ||
      program->graph().nodes.back().footprint.pattern !=
          graph::AccessPattern::Pointwise ||
      program->graph().nodes[1u].footprint.pattern !=
          graph::AccessPattern::Window) {
    std::fprintf(stderr, "virtual window program ok=%u nodes=%zu pattern=%u\n",
                 static_cast<unsigned>(static_cast<bool>(program)),
                 program ? program->graph().nodes.size() : 0u,
                 program && !program->graph().nodes.empty()
                     ? static_cast<unsigned>(
                           program->graph().nodes.back().footprint.pattern)
                     : 255u);
    return 2;
  }
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  std::array<std::int32_t, WindowElements> input_values{};
  for (std::size_t index = 0u; index < input_values.size(); ++index) {
    input_values[index] = value(index);
  }
  if (!input_backing->seed(std::as_bytes(std::span{input_values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::int32_t>(WindowElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(WindowElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  ProductRouteObservation window_route{};
  bool window_run_ok = false;
  if (prepared) {
    ProductRouteScope scope{*device, window_route};
    if (!scope) {
      return 4;
    }
    window_run_ok = static_cast<bool>(prepared->run());
  }
  ResolveProductRoute(window_route, backend, window_run_ok);
  if (!window_run_ok) {
    return 4;
  }
  std::array<std::byte, WindowElements * sizeof(std::int32_t)> observed{};
  if (!output_backing->observe(observed)) {
    return 5;
  }
  for (std::size_t index = 0u; index < WindowElements; ++index) {
    std::int32_t actual = 0;
    std::memcpy(&actual, observed.data() + index * sizeof(actual),
                sizeof(actual));
    if (actual != expected(index)) {
      return 6;
    }
  }
  const Stats stats = prepared->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  // N=53 is the Dedicated Window contract: five canonical pages, three
  // expanded epochs, and one backend-owned Window submission per epoch on
  // Metal (or one queue call containing those records on Vulkan).
  constexpr std::uint64_t ExpectedPhysicalBytes =
      WindowPages * FrameElements * sizeof(std::int32_t);
  constexpr std::uint64_t ExpectedEpochs = (WindowPages + 2u - 1u) / 2u;
  const bool cpu_direct = backend == Backend::Cpu;
  const bool dedicated_window = !cpu_direct && window_route_ok(window_route);
  const std::uint64_t expected_window_queue_calls =
      backend == Backend::Vulkan ? 1u : ExpectedEpochs;
  const std::uint64_t expected_late_pages = WindowPages;
  const std::uint64_t expected_backing_elements = WindowElements;
  const bool receipt_exact =
      cpu_direct ? residency.window_handoff_count == 0u &&
                       residency.window_batch_count == 0u &&
                       residency.window_queue_call_count == 0u
                 : dedicated_window && residency.window_handoff_count == 1u &&
                       residency.window_batch_count == ExpectedEpochs &&
                       residency.window_queue_call_count ==
                           expected_window_queue_calls;
  const bool zero_physical =
      stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u;
  const bool vulkan_physical = backend == Backend::Vulkan &&
                               stats.uploaded_bytes == ExpectedPhysicalBytes &&
                               stats.downloaded_bytes == ExpectedPhysicalBytes;
  const bool transfer_exact = backend == Backend::Vulkan
                                  ? zero_physical || vulkan_physical
                                  : zero_physical;
  if (prepared->plan().residency.page_count != WindowPages ||
      residency.page_count != WindowPages || residency.frame_capacity != 2u ||
      !((cpu_direct || dedicated_window) &&
        residency.page_in_count == WindowPages) ||
      residency.late_page_count != expected_late_pages ||
      residency.prefetch_count != 0u ||
      residency.backing_read_bytes !=
          expected_backing_elements * sizeof(std::int32_t) ||
      residency.page_in_bytes !=
          (cpu_direct ? residency.backing_read_bytes
                      : WindowPages * FrameElements * sizeof(std::int32_t)) ||
      residency.backing_write_bytes != observed.size() || !transfer_exact ||
      (!cpu_direct && stats.command_submits != expected_window_queue_calls) ||
      !receipt_exact) {
    std::fprintf(
        stderr,
        "virtual window backend=%u pages=%llu/%zu frames=%llu loads=%llu "
        "late=%llu prefetch=%llu backing=%llu pagein=%llu write=%llu "
        "upload=%llu download=%llu receipt=%llu/%llu/%llu submits=%llu "
        "peak=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(residency.page_count), WindowPages,
        static_cast<unsigned long long>(residency.frame_capacity),
        static_cast<unsigned long long>(residency.page_in_count),
        static_cast<unsigned long long>(residency.late_page_count),
        static_cast<unsigned long long>(residency.prefetch_count),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(residency.page_in_bytes),
        static_cast<unsigned long long>(residency.backing_write_bytes),
        static_cast<unsigned long long>(stats.uploaded_bytes),
        static_cast<unsigned long long>(stats.downloaded_bytes),
        static_cast<unsigned long long>(residency.window_handoff_count),
        static_cast<unsigned long long>(residency.window_batch_count),
        static_cast<unsigned long long>(residency.window_queue_call_count),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.command_inflight_peak));
    return 7;
  }

  if (const int exact = CheckExactWindow(backend, *device, *program);
      exact != 0) {
    return exact;
  }
  if (const int clip = CheckClipWindow(backend, *device, *input, observed);
      clip != 0) {
    return clip;
  }
  if (const int tier =
          CheckTierWindow(backend, *device, *program, input_values, observed);
      tier != 0) {
    return tier;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
