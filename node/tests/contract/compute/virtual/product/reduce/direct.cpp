#include "local.hpp"

namespace rund_node_test_virtual::product::reduce {

int RunDirectReduce(ReduceFixture &fixture,
                    const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (!fixture.device || !fixture.sum_program || !fixture.output ||
      !fixture.output_backing) {
    return 14;
  }
  auto *device = &*fixture.device;
  const auto *sum_program = &*fixture.sum_program;
  auto *output = &*fixture.output;
  auto &output_backing = fixture.output_backing;
  const auto &values = fixture.values;
  const auto expected_sum = fixture.expected_sum;
  std::uint64_t observed = 0u;
  auto sum_input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(values), ReduceFrameElements * sizeof(std::uint64_t));
  auto sum_input =
      virtual_buffer<std::uint64_t>(ReduceElements, sum_input_backing);
  if (!sum_input_backing->seed(std::as_bytes(std::span{values}))) {
    return 14;
  }
  auto sum = sum_input && output
                 ? virtual_pipeline(*sum_program, *sum_input, *output,
                                    ResidencyConfig{})
                 : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                       Reason::PipelineInvalid);
  ProductRouteObservation sum_obs{};
  Status sum_status = Status::fail(Reason::PipelineInvalid);
  {
    ProductRouteScope sum_scope{*device, sum_obs};
    sum_status = !sum         ? Status::fail(sum.reason())
                 : !sum_scope ? Status::fail(Reason::PipelineInvalid)
                              : sum->run();
  }
  ResolveProductRoute(sum_obs, backend, static_cast<bool>(sum_status));
  if (!sum || !sum_status || !observe_u64(*output_backing, observed) ||
      observed != expected_sum) {
    std::fprintf(stderr,
                 "virtual reduce composed backend=%u observed=%llu "
                 "expected=%llu reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(observed),
                 static_cast<unsigned long long>(expected_sum),
                 static_cast<int>(sum_status.error().size()),
                 sum_status.error().data());
    return 5;
  }
  const Stats sum_stats = sum->stats();
  const ResidencyStats &sum_residency = sum_stats.pipeline.residency;
  const auto &transfer = sum_stats.transfer_submissions;
  std::uint64_t expected_uploaded = 0u;
  std::uint64_t expected_downloaded = 0u;
  bool transfer_exact = false;
  switch (sum_obs.kind) {
  case RouteKind::CpuRolling:
    transfer_exact =
        transfer.host_to_device == 0u && transfer.device_to_host == 0u;
    break;
  case RouteKind::DeviceVsm:
    transfer_exact =
        transfer.host_to_device <= 1u && transfer.device_to_host <= 1u;
    if (transfer_exact) {
      expected_uploaded = transfer.host_to_device * sizeof(values);
      expected_downloaded = transfer.device_to_host * sizeof(std::uint64_t);
    }
    break;
  case RouteKind::AccelRolling:
    transfer_exact = true;
    expected_uploaded =
        ReducePages * ReduceFrameElements * sizeof(std::uint64_t);
    expected_downloaded = ReducePages * sizeof(std::uint64_t);
    break;
  default:
    break;
  }
  if (sum->plan().residency.page_count != ReducePages ||
      sum_residency.page_in_count != ReducePages ||
      sum_residency.page_out_count != 1u ||
      sum_residency.backing_read_bytes != sizeof(values) ||
      sum_residency.backing_write_bytes != sizeof(std::uint64_t) ||
      sum_residency.page_out_bytes != sizeof(std::uint64_t) ||
      !transfer_exact || sum_stats.uploaded_bytes != expected_uploaded ||
      sum_stats.downloaded_bytes != expected_downloaded) {
    return 6;
  }
  if (!sum->run(0u) || !observe_u64(*output_backing, observed) ||
      observed != 0u || sum->stats().pipeline.residency.page_in_count != 0u ||
      sum->stats().pipeline.residency.page_out_count != 1u) {
    return 7;
  }

  auto min_flow = on(*device).input<std::int32_t>(ReduceFrameElements);
  auto min_program =
      std::move(min_flow)
          .branch([](auto values) { return values.reduce(Reduce::Min); })
          .compile();
  std::array<std::int32_t, ReduceElements> signed_values{};
  std::int32_t expected_min = std::numeric_limits<std::int32_t>::max();
  for (std::size_t index = 0u; index < signed_values.size(); ++index) {
    signed_values[index] = static_cast<std::int32_t>(index * 13u % 97u) - 45;
    expected_min = std::min(expected_min, signed_values[index]);
  }
  auto min_input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(signed_values), ReduceFrameElements * sizeof(std::int32_t));
  auto min_output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::int32_t), sizeof(std::int32_t));
  if (!min_program ||
      !min_input_backing->seed(std::as_bytes(std::span{signed_values}))) {
    return 8;
  }
  auto min_input =
      virtual_buffer<std::int32_t>(ReduceElements, min_input_backing);
  auto min_output = virtual_buffer<std::int32_t>(1u, min_output_backing);
  auto minimum =
      min_input && min_output
          ? virtual_pipeline(*min_program, *min_input, *min_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  std::int32_t observed_min = 0;
  if (!minimum || !minimum->run() ||
      !observe_i32(*min_output_backing, observed_min) ||
      observed_min != expected_min ||
      minimum->run(0u).reason() != Reason::ReduceCountZero) {
    return 9;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::reduce
