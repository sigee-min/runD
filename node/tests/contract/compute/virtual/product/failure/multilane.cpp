#include "local.hpp"

#include "src/compute/device/state.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)
#include "src/accel/kernel/fault.hpp"
#endif

#include <array>
#include <cstdio>
#include <cstring>

namespace rund_node_test_virtual::product::failure {

int CheckMultiLaneFailure(FailureFixture &fixture) {
  using namespace rund::compute;
  if (fixture.backend == Backend::Cpu) {
    return 0;
  }
  auto &program = *fixture.program;
  auto &opened = *fixture.device;

  // e+1 fails while the second preallocated lane may already hold e+2.
  // Cancellation must invalidate and complete both Host leases so a retry,
  // the next freeze, and Pool destruction cannot observe a leaked token.
  auto concurrent_input_backing =
      std::make_shared<OffsetFailPersistentBacking>(fixture.seeded);
  auto concurrent_output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto concurrent_input =
      virtual_buffer<std::int32_t>(LogicalElements, concurrent_input_backing);
  auto concurrent_output =
      virtual_buffer<std::int32_t>(LogicalElements, concurrent_output_backing);
  auto concurrent =
      concurrent_input && concurrent_output
          ? virtual_pipeline(program, *concurrent_input, *concurrent_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!concurrent) {
    std::fprintf(stderr, "virtual concurrent construction reason=%u\n",
                 static_cast<unsigned>(concurrent.reason()));
    return 10;
  }
  const Status first = concurrent->run();
  const Stats first_stats = concurrent->stats();
  if (first.reason() != Reason::BackendFailed) {
    std::fprintf(
        stderr,
        "virtual concurrent first reason=%u failed_page=%llu epochs=%llu "
        "page_in=%llu page_out=%llu\n",
        static_cast<unsigned>(first.reason()),
        static_cast<unsigned long long>(
            first_stats.pipeline.residency.failed_page),
        static_cast<unsigned long long>(
            first_stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(
            first_stats.pipeline.residency.page_in_count),
        static_cast<unsigned long long>(
            first_stats.pipeline.residency.page_out_count));
    return 10;
  }
  const Status retry = concurrent->run();
  const Stats retry_stats = concurrent->stats();
  if (!retry) {
    std::fprintf(
        stderr,
        "virtual concurrent retry reason=%u success=%u failed_page=%llu "
        "epochs=%llu page_in=%llu page_out=%llu\n",
        static_cast<unsigned>(retry.reason()),
        static_cast<unsigned>(retry.ok()),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.failed_page),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.page_in_count),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.page_out_count));
    return 10;
  }

  constexpr std::size_t graph_page_elements = 16u;
  constexpr std::size_t graph_elements = 137u;
  constexpr std::size_t graph_page_bytes =
      graph_page_elements * sizeof(std::uint64_t);
  std::array<std::uint64_t, graph_elements> graph_values{};
  std::uint64_t graph_expected = graph_elements;
  for (std::size_t index = 0u; index < graph_values.size(); ++index) {
    graph_values[index] = index * 7u + 3u;
    graph_expected += graph_values[index];
  }
  auto graph_program =
      on(opened)
          .map<std::uint64_t>("virtual-product-graph-backing-retry",
                              graph_page_elements,
                              [](auto value) { return value + 1u; })
          .reduce(Reduce::Sum)
          .compile();
  auto graph_input_backing = std::make_shared<OffsetFailPersistentBacking>(
      std::as_bytes(std::span{graph_values}), graph_page_bytes * 2u);
  auto graph_output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::uint64_t), sizeof(std::uint64_t));
  auto graph_input =
      virtual_buffer<std::uint64_t>(graph_elements, graph_input_backing);
  auto graph_output = virtual_buffer<std::uint64_t>(1u, graph_output_backing);
  auto graph =
      graph_program && graph_input && graph_output
          ? virtual_pipeline(*graph_program, *graph_input, *graph_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                Reason::PipelineInvalid);
  const Status graph_failed =
      graph ? graph->run() : Status::fail(graph.reason());
  const Stats graph_failed_stats = graph ? graph->stats() : Stats{};
  const Status graph_retry =
      graph ? graph->run() : Status::fail(graph.reason());
  std::array<std::byte, sizeof(std::uint64_t)> graph_observed_bytes{};
  std::uint64_t graph_observed = 0u;
  if (!graph || graph_failed.reason() != Reason::BackendFailed ||
      graph_failed_stats.pipeline.residency.failed_page != 2u || !graph_retry ||
      !graph_output_backing->observe(graph_observed_bytes)) {
    std::fprintf(stderr,
                 "virtual graph backing retry backend=%u prepared=%u failed=%u "
                 "failed_page=%llu retry=%u observations=%llu\n",
                 static_cast<unsigned>(fixture.backend), graph ? 1u : 0u,
                 static_cast<unsigned>(graph_failed.reason()),
                 static_cast<unsigned long long>(
                     graph_failed_stats.pipeline.residency.failed_page),
                 static_cast<unsigned>(graph_retry.reason()),
                 static_cast<unsigned long long>(
                     graph_output_backing->facts().observation_count));
    return 11;
  }
  std::memcpy(&graph_observed, graph_observed_bytes.data(),
              sizeof(graph_observed));
  if (graph_observed != graph_expected || !graph->run()) {
    return 12;
  }
#if !defined(RUND_NODE_TEST_BACKEND_CPU)
  if (fixture.backend == Backend::Metal) {
    const std::shared_ptr<detail::DeviceState> &device_state =
        detail::DeviceAccess::state(opened);
    detail::AccelDeviceState *const native =
        device_state == nullptr ? nullptr : detail::accel_device(*device_state);
    const BackingFacts before_download_failure = graph_output_backing->facts();
    if (native == nullptr ||
        !rund::node::accel::detail::InjectNativeDownloadFailureOnce(
            native->pick)) {
      return 13;
    }
    const Status download_failed = graph->run();
    const BackingFacts after_download_failure = graph_output_backing->facts();
    const Status download_retry = graph->run();
    if (download_failed.reason() != Reason::TransferInvalid ||
        after_download_failure.write_count !=
            before_download_failure.write_count ||
        after_download_failure.write_failure_count !=
            before_download_failure.write_failure_count ||
        !download_retry ||
        !graph_output_backing->observe(graph_observed_bytes)) {
      std::fprintf(
          stderr,
          "virtual graph download failure reason=%u writes=%llu/%llu "
          "write_failures=%llu/%llu retry=%u observations=%llu\n",
          static_cast<unsigned>(download_failed.reason()),
          static_cast<unsigned long long>(after_download_failure.write_count),
          static_cast<unsigned long long>(before_download_failure.write_count),
          static_cast<unsigned long long>(
              after_download_failure.write_failure_count),
          static_cast<unsigned long long>(
              before_download_failure.write_failure_count),
          static_cast<unsigned>(download_retry.reason()),
          static_cast<unsigned long long>(
              graph_output_backing->facts().observation_count));
      return 14;
    }
    std::memcpy(&graph_observed, graph_observed_bytes.data(),
                sizeof(graph_observed));
    if (graph_observed != graph_expected || !graph->run()) {
      return 15;
    }
  }
#endif
  return 0;
}

} // namespace rund_node_test_virtual::product::failure
