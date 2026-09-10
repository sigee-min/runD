#include "local.hpp"

namespace rund_node_test_virtual::product::reduce {

int RunBasicReduce(ReduceFixture &fixture,
                   const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (!fixture.device || !fixture.input || !fixture.output ||
      !fixture.mapped_reduce_program || !fixture.input_backing ||
      !fixture.output_backing) {
    return 4;
  }
  const auto &values = fixture.values;
  const auto expected_sum = fixture.expected_sum;
  auto &output_backing = fixture.output_backing;
  auto &input = *fixture.input;
  auto &output = *fixture.output;
  const auto &mapped_reduce_program = *fixture.mapped_reduce_program;
  auto mapped_reduce =
      virtual_pipeline(mapped_reduce_program, input, output, ResidencyConfig{});
  std::uint64_t observed = 0u;
  const std::uint64_t expected_mapped_sum = expected_sum + ReduceElements;
  const Status mapped_status = mapped_reduce
                                   ? mapped_reduce->run()
                                   : Status::fail(mapped_reduce.reason());
  if (!mapped_reduce_program || !mapped_reduce || !mapped_status ||
      mapped_reduce->plan().residency.frame_capacity != 2u ||
      !observe_u64(*output_backing, observed) ||
      observed != expected_mapped_sum) {
    std::fprintf(stderr,
                 "virtual mapped reduce backend=%u observed=%llu "
                 "expected=%llu reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(observed),
                 static_cast<unsigned long long>(expected_mapped_sum),
                 static_cast<int>(mapped_status.error().size()),
                 mapped_status.error().data());
    return 4;
  }
  constexpr std::uint64_t ActiveMapped = 71u;
  std::uint64_t expected_active_mapped = ActiveMapped;
  for (std::size_t index = 0u; index < ActiveMapped; ++index) {
    expected_active_mapped += values[index];
  }
  if (!mapped_reduce->run(ActiveMapped) ||
      !observe_u64(*output_backing, observed) ||
      observed != expected_active_mapped || !mapped_reduce->run(ActiveMapped) ||
      !observe_u64(*output_backing, observed) ||
      observed != expected_active_mapped ||
      mapped_reduce->stats().buffer_allocations != 0u) {
    return 10;
  }
  const PipelinePlan graph_plan = mapped_reduce->plan();
  const MemoryStats graph_memory = mapped_reduce->memory();
  if (!mapped_reduce->begin_samples()) {
    return 11;
  }
  node_compute_allocation::Start();
  bool graph_overlap_observed = false;
  bool graph_h2d_overlap_observed = false;
  bool graph_d2h_overlap_observed = false;
  Status graph_warm = Status::success();
  for (std::size_t attempt = 0u; attempt < GraphWarmRuns; ++attempt) {
    graph_warm = mapped_reduce->run(ActiveMapped);
    if (!graph_warm) {
      break;
    }
    graph_overlap_observed =
        graph_overlap_observed ||
        mapped_reduce->stats().pipeline.residency.overlap_ns != 0u;
    graph_h2d_overlap_observed =
        graph_h2d_overlap_observed ||
        mapped_reduce->stats().pipeline.residency.h2d_overlap_ns != 0u;
    graph_d2h_overlap_observed =
        graph_d2h_overlap_observed ||
        mapped_reduce->stats().pipeline.residency.d2h_overlap_ns != 0u;
  }
  node_compute_allocation::Stop();
  if (!mapped_reduce->end_samples()) {
    return 11;
  }
  const Stats graph_stats = mapped_reduce->stats();
  const ResidencyStats &graph_residency = graph_stats.pipeline.residency;
  const std::uint64_t active_pages =
      (ActiveMapped + ReduceFrameElements - 1u) / ReduceFrameElements;
  const std::uint64_t active_batches =
      (active_pages + mapped_reduce->plan().residency.frame_capacity - 1u) /
      mapped_reduce->plan().residency.frame_capacity;
  const bool graph_device_vsm = backend != Backend::Cpu &&
                                graph_residency.window_handoff_count == 1u &&
                                graph_residency.window_batch_count == 1u &&
                                graph_residency.window_queue_call_count == 1u;
  const std::uint64_t expected_graph_epochs =
      graph_device_vsm ? active_batches : active_batches * 2u;
  if (!graph_warm ||
      !(graph_device_vsm ||
        allocation_boundary_exact(backend, node_compute_allocation::Count())) ||
      mapped_reduce->plan() != graph_plan ||
      !same_fixed_memory(mapped_reduce->memory(), graph_memory) ||
      !graph_residency.samples_allocation_free(GraphWarmRuns) ||
      !observe_u64(*output_backing, observed) ||
      observed != expected_active_mapped ||
      graph_residency.epoch_count != expected_graph_epochs ||
      graph_overlap_observed !=
          (backend != Backend::Cpu && !graph_device_vsm) ||
      graph_h2d_overlap_observed !=
          (backend != Backend::Cpu && !graph_device_vsm) ||
      graph_d2h_overlap_observed !=
          (backend != Backend::Cpu && !graph_device_vsm) ||
      !graph_residency.directional_overlap_exact() ||
      graph_stats.buffer_allocations != 0u) {
    std::fprintf(
        stderr,
        "virtual mapped reduce warm backend=%u reason=%.*s alloc=%llu "
        "epochs=%llu expected_epochs=%llu overlap=%u overlap_ns=%llu "
        "h2d_overlap_ns=%llu d2h_overlap_ns=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<int>(graph_warm.error().size()), graph_warm.error().data(),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<unsigned long long>(graph_residency.epoch_count),
        static_cast<unsigned long long>(expected_graph_epochs),
        graph_overlap_observed ? 1u : 0u,
        static_cast<unsigned long long>(graph_residency.overlap_ns),
        static_cast<unsigned long long>(graph_residency.h2d_overlap_ns),
        static_cast<unsigned long long>(graph_residency.d2h_overlap_ns));
    return 11;
  }

  fixture.mapped_reduce.emplace(std::move(mapped_reduce).value());
  return 0;
}

} // namespace rund_node_test_virtual::product::reduce
