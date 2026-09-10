#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product::window_detail {

int CheckTierWindow(const rund::compute::Backend backend,
                    rund::compute::Device &device, const WindowProgram &program,
                    const std::span<const std::int32_t> input_values,
                    const std::span<std::byte> observed) {
  using namespace rund::compute;
  constexpr std::uint64_t ExpectedEpochs = (WindowPages + 2u - 1u) / 2u;
  if (backend != Backend::Cpu) {
    const std::uint64_t tier_queue_calls =
        backend == Backend::Vulkan ? 1u : ExpectedEpochs;
    auto tier_input = std::make_shared<MemoryVirtualBacking>(
        WindowElements * sizeof(std::int32_t),
        FrameElements * sizeof(std::int32_t), VirtualBackingTier::Persistent);
    auto tier_output = std::make_shared<MemoryVirtualBacking>(
        WindowElements * sizeof(std::int32_t),
        FrameElements * sizeof(std::int32_t));
    if (!tier_input->seed(std::as_bytes(input_values))) {
      return 12;
    }
    auto tier_input_buffer =
        virtual_buffer<std::int32_t>(WindowElements, tier_input);
    auto tier_output_buffer =
        virtual_buffer<std::int32_t>(WindowElements, tier_output);
    auto tier_prepared =
        tier_input_buffer && tier_output_buffer
            ? virtual_pipeline(program, *tier_input_buffer, *tier_output_buffer,
                               ResidencyConfig{})
            : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                  Reason::PipelineInvalid);
    ProductRouteObservation tier_route{};
    bool tier_run_ok = false;
    if (tier_prepared) {
      ProductRouteScope scope{device, tier_route};
      if (!scope) {
        return 13;
      }
      tier_run_ok = static_cast<bool>(tier_prepared->run());
    }
    ResolveProductRoute(tier_route, backend, tier_run_ok);
    if (!tier_run_ok || !tier_output->observe(observed)) {
      return 13;
    }
    for (std::size_t index = 0u; index < WindowElements; ++index) {
      std::int32_t actual = 0;
      std::memcpy(&actual, observed.data() + index * sizeof(actual),
                  sizeof(actual));
      if (actual != expected(index)) {
        return 14;
      }
    }
    const Stats tier_stats = tier_prepared->stats();
    const ResidencyStats &tier_residency = tier_stats.pipeline.residency;
    if (!window_route_ok(tier_route) ||
        tier_residency.page_count != WindowPages ||
        tier_residency.page_in_count != WindowPages ||
        tier_residency.late_page_count != WindowPages ||
        tier_residency.prefetch_count != 0u ||
        tier_residency.window_handoff_count != 1u ||
        tier_residency.window_batch_count != ExpectedEpochs ||
        tier_residency.window_queue_call_count != tier_queue_calls ||
        tier_residency.backing_read_bytes !=
            WindowElements * sizeof(std::int32_t) ||
        tier_stats.command_submits != tier_queue_calls) {
      return 15;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::window_detail
