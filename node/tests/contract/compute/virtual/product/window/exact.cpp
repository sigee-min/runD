#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product::window_detail {

int CheckExactWindow(const rund::compute::Backend backend,
                     rund::compute::Device &device,
                     const WindowProgram &program) {
  using namespace rund::compute;
  // N=48 is an exact canonical P=12-page input, but F=16 and R=2 still make
  // the last expanded frame a clipped/fill frame. The public accelerator
  // route must therefore retain the canonical tail as a full page, read each
  // canonical page once, and promote each expanded output frame exactly once
  // through the Dedicated Window owner.
  if (backend != Backend::Cpu) {
    constexpr std::size_t ExactWindowElements = 48u;
    constexpr std::size_t ExactWindowPages =
        (ExactWindowElements + PayloadElements - 1u) / PayloadElements;
    constexpr std::size_t ExactEpochs = (ExactWindowPages + 2u - 1u) / 2u;
    constexpr std::uint64_t ExactLogicalBytes =
        ExactWindowElements * sizeof(std::int32_t);
    constexpr std::uint64_t ExactPhysicalBytes =
        ExactWindowPages * FrameElements * sizeof(std::int32_t);
    auto exact_input_backing = std::make_shared<MemoryVirtualBacking>(
        ExactLogicalBytes, FrameElements * sizeof(std::int32_t));
    auto exact_output_backing = std::make_shared<MemoryVirtualBacking>(
        ExactLogicalBytes, FrameElements * sizeof(std::int32_t));
    std::array<std::int32_t, ExactWindowElements> exact_input_values{};
    for (std::size_t index = 0u; index < exact_input_values.size(); ++index) {
      exact_input_values[index] = value(index);
    }
    if (!exact_input_backing->seed(
            std::as_bytes(std::span{exact_input_values}))) {
      return 16;
    }
    auto exact_input =
        virtual_buffer<std::int32_t>(ExactWindowElements, exact_input_backing);
    auto exact_output =
        virtual_buffer<std::int32_t>(ExactWindowElements, exact_output_backing);
    auto exact_prepared =
        exact_input && exact_output
            ? virtual_pipeline(program, *exact_input, *exact_output,
                               ResidencyConfig{})
            : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                  Reason::PipelineInvalid);
    std::array<std::byte, ExactLogicalBytes> exact_observed{};
    ProductRouteObservation exact_route{};
    bool exact_run_ok = false;
    if (exact_prepared) {
      ProductRouteScope scope{device, exact_route};
      if (!scope) {
        return 17;
      }
      exact_run_ok = static_cast<bool>(exact_prepared->run());
    }
    ResolveProductRoute(exact_route, backend, exact_run_ok);
    if (!exact_run_ok || !exact_output_backing->observe(exact_observed)) {
      return 17;
    }
    for (std::size_t index = 0u; index < ExactWindowElements; ++index) {
      std::int32_t actual = 0;
      std::memcpy(&actual, exact_observed.data() + index * sizeof(actual),
                  sizeof(actual));
      if (actual != expected_for(index, ExactWindowElements)) {
        return 18;
      }
    }
    const Stats exact_stats = exact_prepared->stats();
    const ResidencyStats &exact_residency = exact_stats.pipeline.residency;
    const BackingFacts exact_input_facts = exact_input_backing->facts();
    const BackingFacts exact_output_facts = exact_output_backing->facts();
    const bool exact_vulkan_transfer =
        backend == Backend::Vulkan &&
        exact_stats.uploaded_bytes == ExactPhysicalBytes &&
        exact_stats.downloaded_bytes == ExactPhysicalBytes;
    const bool exact_transfer =
        exact_stats.uploaded_bytes == 0u && exact_stats.downloaded_bytes == 0u;
    const std::uint64_t exact_window_queue_calls =
        backend == Backend::Vulkan ? 1u : ExactEpochs;
    if (!window_route_ok(exact_route) ||
        exact_prepared->plan().residency.page_count != ExactWindowPages ||
        exact_residency.page_count != ExactWindowPages ||
        exact_residency.frame_capacity != 2u ||
        exact_residency.window_handoff_count != 1u ||
        exact_residency.window_batch_count != ExactEpochs ||
        exact_residency.window_queue_call_count != exact_window_queue_calls ||
        exact_residency.page_in_count != ExactWindowPages ||
        exact_residency.page_in_bytes != ExactPhysicalBytes ||
        exact_residency.backing_read_bytes != ExactLogicalBytes ||
        exact_residency.backing_write_bytes != ExactLogicalBytes ||
        exact_residency.page_out_count != ExactWindowPages ||
        exact_residency.cache_hit_count != 0u ||
        exact_residency.late_page_count != ExactWindowPages ||
        exact_residency.prefetch_count != 0u ||
        exact_input_facts.read_count != ExactWindowPages ||
        exact_input_facts.read_bytes != ExactLogicalBytes ||
        exact_output_facts.write_count != ExactWindowPages ||
        exact_output_facts.write_bytes != ExactLogicalBytes ||
        exact_output_facts.partial_write_bytes != 0u ||
        exact_stats.command_submits != exact_window_queue_calls ||
        (!exact_transfer && !exact_vulkan_transfer)) {
      std::fprintf(
          stderr,
          "exact window backend=%u pages=%llu frames=%llu loads=%llu "
          "pagein=%llu/%llu read=%llu/%llu writes=%llu/%llu "
          "hits=%llu late=%llu prefetch=%llu receipt=%llu/%llu/%llu "
          "submits=%llu upload=%llu download=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(exact_residency.page_count),
          static_cast<unsigned long long>(exact_residency.frame_capacity),
          static_cast<unsigned long long>(exact_residency.page_in_count),
          static_cast<unsigned long long>(exact_residency.page_in_bytes),
          static_cast<unsigned long long>(ExactPhysicalBytes),
          static_cast<unsigned long long>(exact_input_facts.read_bytes),
          static_cast<unsigned long long>(exact_input_facts.read_count),
          static_cast<unsigned long long>(exact_output_facts.write_bytes),
          static_cast<unsigned long long>(exact_output_facts.write_count),
          static_cast<unsigned long long>(exact_residency.cache_hit_count),
          static_cast<unsigned long long>(exact_residency.late_page_count),
          static_cast<unsigned long long>(exact_residency.prefetch_count),
          static_cast<unsigned long long>(exact_residency.window_handoff_count),
          static_cast<unsigned long long>(exact_residency.window_batch_count),
          static_cast<unsigned long long>(
              exact_residency.window_queue_call_count),
          static_cast<unsigned long long>(exact_stats.command_submits),
          static_cast<unsigned long long>(exact_stats.uploaded_bytes),
          static_cast<unsigned long long>(exact_stats.downloaded_bytes));
      return 19;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::window_detail
