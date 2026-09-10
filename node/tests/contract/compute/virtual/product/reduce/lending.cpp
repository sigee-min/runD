#include "local.hpp"

namespace rund_node_test_virtual::product::reduce {

int RunLendingReduce(ReduceFixture &fixture,
                     const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (!fixture.device || !fixture.input || !fixture.sum_program ||
      !fixture.mapped_reduce_program || !fixture.input_backing) {
    return 40;
  }
  auto *input = &*fixture.input;
  const auto &values = fixture.values;
  const auto expected_mapped_sum = fixture.expected_sum + ReduceElements;
  const auto *sum_program = &*fixture.sum_program;
  const auto *mapped_reduce_program = &*fixture.mapped_reduce_program;
  std::uint64_t observed = 0u;
  {
    std::uint64_t expected_active_sum = 0u;
    for (std::size_t index = 0u; index < ActiveMapped; ++index) {
      expected_active_sum += values[index];
    }
    // Populate the service-aware cache in this cohort. The preceding native
    // DeviceVsm and sequence cohorts do not promise to retain these rows.
    DeviceVsmBypassScope lending_route{*fixture.device};
    auto lending_graph_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto lending_graph_output =
        virtual_buffer<std::uint64_t>(1u, lending_graph_output_backing);
    auto lending_graph =
        lending_graph_output
            ? virtual_pipeline(*mapped_reduce_program, *input,
                               *lending_graph_output, ResidencyConfig{})
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    if (!lending_route || !lending_graph || !lending_graph->run(ActiveMapped) ||
        !observe_u64(*lending_graph_output_backing, observed) ||
        observed != expected_active_sum + ActiveMapped) {
      return 44;
    }
    // The mapped Graph layout owns Intermediate/Control pages while the direct
    // Reduce layout does not. Their typed input frame class is nevertheless
    // identical, so the Registry must lend the same physical arena and the
    // first direct run must observe cross-layout execution hits.
    auto cross_layout_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto cross_layout_output =
        virtual_buffer<std::uint64_t>(1u, cross_layout_output_backing);
    auto cross_layout_sum =
        cross_layout_output
            ? virtual_pipeline(*sum_program, *input, *cross_layout_output,
                               ResidencyConfig{})
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    const Status cross_layout_status =
        cross_layout_sum ? cross_layout_sum->run(ActiveMapped)
                         : Status::fail(cross_layout_sum.reason());
    const ResidencyStats cross_layout_residency =
        cross_layout_sum ? cross_layout_sum->stats().pipeline.residency
                         : ResidencyStats{};
    if (!cross_layout_status ||
        !observe_u64(*cross_layout_output_backing, observed) ||
        observed != expected_active_sum ||
        cross_layout_residency.cache_hit_count == 0u ||
        cross_layout_residency.backing_read_bytes >=
            ActiveMapped * sizeof(std::uint64_t)) {
      std::fprintf(
          stderr,
          "virtual cross-layout lending backend=%u reason=%.*s observed=%llu "
          "expected=%llu hits=%llu backing=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<int>(cross_layout_status.error().size()),
          cross_layout_status.error().data(),
          static_cast<unsigned long long>(observed),
          static_cast<unsigned long long>(expected_active_sum),
          static_cast<unsigned long long>(
              cross_layout_residency.cache_hit_count),
          static_cast<unsigned long long>(
              cross_layout_residency.backing_read_bytes));
      return 13;
    }
  }
  {
    // Cross-K lending must preserve the canonical arena bank stride C. The
    // first pipeline creates C=4 input banks; K=2 Graph and Direct borrowers
    // consume only exact bank-region prefixes. Reconstructing bank 1 as first+K
    // would alias the bank-0 tail and either report false hits or execute wrong
    // bytes.
    auto cross_k_device = open(rund::node::test_contract::target_for(backend));
    if (!cross_k_device) {
      return 40;
    }
    DeviceVsmBypassScope cross_k_lending_route{*cross_k_device};
    if (!cross_k_lending_route) {
      return 40;
    }
    auto cross_k_graph_program =
        on(*cross_k_device)
            .map<std::uint64_t>("virtual-product-reduce-cross-k",
                                ReduceFrameElements,
                                [](auto value) { return value + 1u; })
            // Keep this cache-arena lending oracle on the service-aware Graph
            // route. The exact DeviceVsm Map-to-Reduce contract is covered
            // above; this semantic identity stage deliberately lies outside its
            // single AddWrapU64Immediate authority.
            .map("virtual-product-reduce-cross-k-identity",
                 [](auto value) { return value ^ 0u; })
            .reduce(Reduce::Sum)
            .compile();
    auto cross_k_sum_flow =
        on(*cross_k_device).input<std::uint64_t>(ReduceFrameElements);
    auto cross_k_sum_program =
        std::move(cross_k_sum_flow)
            .branch([](auto values) { return values.reduce(Reduce::Sum); })
            .compile();
    auto cross_k_graph_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto cross_k_sum_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto cross_k_graph_output =
        virtual_buffer<std::uint64_t>(1u, cross_k_graph_backing);
    auto cross_k_sum_output =
        virtual_buffer<std::uint64_t>(1u, cross_k_sum_backing);
    constexpr std::uint64_t GraphFrameAdmissionBytes =
        2u * (ReduceFrameElements * sizeof(std::uint64_t) * 2u +
              sizeof(std::uint64_t) * 2u);
    constexpr std::uint64_t DirectFrameAdmissionBytes =
        2u *
        (ReduceFrameElements * sizeof(std::uint64_t) + sizeof(std::uint64_t));
    const ResidencyConfig cross_k_large{
        .device_resident_bytes = GraphFrameAdmissionBytes * 4u,
        .host_resident_bytes = GraphFrameAdmissionBytes * 4u,
    };
    const ResidencyConfig cross_k_small{
        .device_resident_bytes = DirectFrameAdmissionBytes * 2u,
        .host_resident_bytes = DirectFrameAdmissionBytes * 2u,
    };
    const ResidencyConfig cross_k_graph_small{
        .device_resident_bytes = GraphFrameAdmissionBytes * 2u,
        .host_resident_bytes = GraphFrameAdmissionBytes * 2u,
    };
    auto cross_k_graph =
        cross_k_graph_program && cross_k_graph_output
            ? virtual_pipeline(*cross_k_graph_program, *input,
                               *cross_k_graph_output, cross_k_large)
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    constexpr std::uint64_t CrossKLargeActive = ReduceFrameElements * 8u;
    const Status cross_k_graph_status =
        cross_k_graph ? cross_k_graph->run(CrossKLargeActive)
                      : Status::fail(cross_k_graph.reason());
    // The same Graph at K=2 must borrow every compatible C=4 execution arena,
    // not only Input.  Its second batch executes physical bank 1 at stride C;
    // any reconstruction as first+K aliases the bank-0 tail and breaks the
    // Intermediate or Output authority/buffer identity.
    auto cross_k_small_graph_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto cross_k_small_graph_output =
        virtual_buffer<std::uint64_t>(1u, cross_k_small_graph_backing);
    auto cross_k_small_graph =
        cross_k_graph_program && cross_k_small_graph_output
            ? virtual_pipeline(*cross_k_graph_program, *input,
                               *cross_k_small_graph_output, cross_k_graph_small)
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    constexpr std::uint64_t CrossKSmallGraphActive = ReduceFrameElements * 4u;
    std::uint64_t expected_cross_k_graph = CrossKSmallGraphActive;
    for (std::size_t index = 0u; index < CrossKSmallGraphActive; ++index) {
      expected_cross_k_graph += values[index];
    }
    const Status cross_k_small_graph_status =
        cross_k_small_graph ? cross_k_small_graph->run(CrossKSmallGraphActive)
                            : Status::fail(cross_k_small_graph.reason());
    const ResidencyStats cross_k_small_graph_residency =
        cross_k_small_graph ? cross_k_small_graph->stats().pipeline.residency
                            : ResidencyStats{};
    // The accelerator's speculative second tranche retains a Host fallback
    // because its earlier Device probe would not be a pin. The later execution
    // lease must nevertheless prove four Device hits and zero promotions; the
    // conservative backing read is producer evidence, not a page-in relabel.
    const std::uint64_t cross_k_small_graph_expected_backing =
        backend == Backend::Cpu
            ? 0u
            : ReduceFrameElements * 2u * sizeof(std::uint64_t);
    std::uint64_t observed_cross_k_graph = 0u;
    if (!cross_k_small_graph || !cross_k_small_graph_status ||
        cross_k_small_graph->plan().residency.frame_capacity != 2u ||
        !observe_u64(*cross_k_small_graph_backing, observed_cross_k_graph) ||
        observed_cross_k_graph != expected_cross_k_graph ||
        cross_k_small_graph_residency.cache_hit_count != 4u ||
        cross_k_small_graph_residency.page_in_count != 0u ||
        cross_k_small_graph_residency.backing_read_bytes !=
            cross_k_small_graph_expected_backing) {
      std::fprintf(
          stderr,
          "virtual graph cross-K lending backend=%u reason=%.*s observed=%llu "
          "expected=%llu k=%llu hits=%llu page_in=%llu backing=%llu "
          "expected_backing=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<int>(cross_k_small_graph_status.error().size()),
          cross_k_small_graph_status.error().data(),
          static_cast<unsigned long long>(observed_cross_k_graph),
          static_cast<unsigned long long>(expected_cross_k_graph),
          static_cast<unsigned long long>(
              cross_k_small_graph
                  ? cross_k_small_graph->plan().residency.frame_capacity
                  : 0u),
          static_cast<unsigned long long>(
              cross_k_small_graph_residency.cache_hit_count),
          static_cast<unsigned long long>(
              cross_k_small_graph_residency.page_in_count),
          static_cast<unsigned long long>(
              cross_k_small_graph_residency.backing_read_bytes),
          static_cast<unsigned long long>(
              cross_k_small_graph_expected_backing));
      return 41;
    }
    auto cross_k_sum =
        cross_k_sum_program && cross_k_sum_output
            ? virtual_pipeline(*cross_k_sum_program, *input,
                               *cross_k_sum_output, cross_k_small)
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    constexpr std::uint64_t CrossKSmallActive = ReduceFrameElements * 4u;
    std::uint64_t expected_cross_k = 0u;
    for (std::size_t index = 0u; index < CrossKSmallActive; ++index) {
      expected_cross_k += values[index];
    }
    const Status cross_k_status = cross_k_sum
                                      ? cross_k_sum->run(CrossKSmallActive)
                                      : Status::fail(cross_k_sum.reason());
    const ResidencyStats cross_k_residency =
        cross_k_sum ? cross_k_sum->stats().pipeline.residency
                    : ResidencyStats{};
    // The K=2 Graph above refreshes pages 0..3 through the borrowed C=4 arena;
    // the subsequent Direct consumer must see all four execution hits.
    const std::uint64_t cross_k_expected_backing = 0u;
    if (!cross_k_device || !cross_k_graph_program || !cross_k_sum_program ||
        !cross_k_graph || !cross_k_sum || !cross_k_graph_status ||
        !cross_k_status ||
        cross_k_graph->plan().residency.frame_capacity != 4u ||
        cross_k_sum->plan().residency.frame_capacity != 2u ||
        !observe_u64(*cross_k_sum_backing, observed) ||
        observed != expected_cross_k ||
        cross_k_residency.cache_hit_count != 4u ||
        cross_k_residency.backing_read_bytes != cross_k_expected_backing) {
      std::fprintf(
          stderr,
          "virtual cross-K lending backend=%u graph_reason=%.*s reason=%.*s "
          "graph_k=%llu direct_k=%llu observed=%llu expected=%llu hits=%llu "
          "backing=%llu expected_backing=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<int>(cross_k_graph_status.error().size()),
          cross_k_graph_status.error().data(),
          static_cast<int>(cross_k_status.error().size()),
          cross_k_status.error().data(),
          static_cast<unsigned long long>(
              cross_k_graph ? cross_k_graph->plan().residency.frame_capacity
                            : 0u),
          static_cast<unsigned long long>(
              cross_k_sum ? cross_k_sum->plan().residency.frame_capacity : 0u),
          static_cast<unsigned long long>(observed),
          static_cast<unsigned long long>(expected_cross_k),
          static_cast<unsigned long long>(cross_k_residency.cache_hit_count),
          static_cast<unsigned long long>(cross_k_residency.backing_read_bytes),
          static_cast<unsigned long long>(cross_k_expected_backing));
      return 40;
    }

    // The U64 C=4 Graph above owns 512-byte Input/Intermediate banks. A second
    // U64 Graph with half-sized pages projects those raw owners as C=8 semantic
    // views while executing only K=2 prefixes. This is an actual CPU/Metal
    // kernel run across page geometry, not only a Pool handle identity check.
    // Switching back to the already prepared C=4 pipeline proves that view
    // activation evicts stale clean rows without poisoning either owner.
    constexpr std::uint64_t ReblockedFrameElements = ReduceFrameElements / 2u;
    auto reblocked_input_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(values), ReblockedFrameElements * sizeof(std::uint64_t));
    auto reblocked_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto reblocked_input =
        virtual_buffer<std::uint64_t>(ReduceElements, reblocked_input_backing);
    auto reblocked_output =
        virtual_buffer<std::uint64_t>(1u, reblocked_output_backing);
    auto reblocked_program =
        on(*cross_k_device)
            .map<std::uint64_t>("virtual-product-reduce-reblocked-view",
                                ReblockedFrameElements,
                                [](auto value) { return value + 1u; })
            .map("virtual-product-reduce-reblocked-view-identity",
                 [](auto value) { return value ^ 0u; })
            .reduce(Reduce::Sum)
            .compile();
    constexpr std::uint64_t ReblockedFrameAdmissionBytes =
        2u * (ReblockedFrameElements * sizeof(std::uint64_t) * 2u +
              sizeof(std::uint64_t) * 2u);
    const ResidencyConfig reblocked_config{
        .device_resident_bytes = ReblockedFrameAdmissionBytes * 2u,
        .host_resident_bytes = ReblockedFrameAdmissionBytes * 2u,
    };
    auto reblocked =
        reblocked_program && reblocked_input && reblocked_output &&
                reblocked_input_backing->seed(std::as_bytes(std::span{values}))
            ? virtual_pipeline(*reblocked_program, *reblocked_input,
                               *reblocked_output, reblocked_config)
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    constexpr std::uint64_t ReblockedActive = ReduceFrameElements * 4u;
    std::uint64_t expected_reblocked = ReblockedActive;
    for (std::size_t index = 0u; index < ReblockedActive; ++index) {
      expected_reblocked += values[index];
    }
    const Status reblocked_status = reblocked
                                        ? reblocked->run(ReblockedActive)
                                        : Status::fail(reblocked.reason());
    std::uint64_t observed_reblocked = 0u;
    const ResidencyStats reblocked_residency =
        reblocked ? reblocked->stats().pipeline.residency : ResidencyStats{};
    if (!reblocked || !reblocked_status ||
        reblocked->plan().residency.frame_capacity != 2u ||
        !observe_u64(*reblocked_output_backing, observed_reblocked) ||
        observed_reblocked != expected_reblocked ||
        reblocked_residency.eviction_count == 0u) {
      std::fprintf(
          stderr,
          "virtual reblocked view backend=%u reason=%.*s observed=%llu "
          "expected=%llu k=%llu evictions=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<int>(reblocked_status.error().size()),
          reblocked_status.error().data(),
          static_cast<unsigned long long>(observed_reblocked),
          static_cast<unsigned long long>(expected_reblocked),
          static_cast<unsigned long long>(
              reblocked ? reblocked->plan().residency.frame_capacity : 0u),
          static_cast<unsigned long long>(reblocked_residency.eviction_count));
      return 42;
    }
    const Status u64_after_reblocked = cross_k_graph->run(CrossKLargeActive);
    std::uint64_t expected_u64_after_reblocked = CrossKLargeActive;
    for (std::size_t index = 0u; index < CrossKLargeActive; ++index) {
      expected_u64_after_reblocked += values[index];
    }
    std::uint64_t observed_u64_after_reblocked = 0u;
    if (!u64_after_reblocked ||
        !observe_u64(*cross_k_graph_backing, observed_u64_after_reblocked) ||
        observed_u64_after_reblocked != expected_u64_after_reblocked ||
        cross_k_graph->stats().pipeline.residency.eviction_count == 0u) {
      return 43;
    }
    const PipelinePlan reblocked_plan = reblocked->plan();
    const MemoryStats reblocked_memory = reblocked->memory();
    if (!reblocked->begin_samples()) {
      return 44;
    }
    node_compute_allocation::Start();
    Status reblocked_warm = Status::success();
    for (std::size_t attempt = 0u; attempt < GraphWarmRuns; ++attempt) {
      reblocked_warm = reblocked->run(ReblockedActive);
      if (!reblocked_warm) {
        break;
      }
    }
    node_compute_allocation::Stop();
    if (!reblocked->end_samples()) {
      return 44;
    }
    if (!reblocked_warm ||
        !allocation_boundary_exact(backend, node_compute_allocation::Count()) ||
        reblocked->plan() != reblocked_plan ||
        !same_fixed_memory(reblocked->memory(), reblocked_memory) ||
        !reblocked->stats().pipeline.residency.samples_allocation_free(
            GraphWarmRuns) ||
        !observe_u64(*reblocked_output_backing, observed_reblocked) ||
        observed_reblocked != expected_reblocked ||
        reblocked->stats().buffer_allocations != 0u) {
      return 44;
    }
    auto delayed_input_backing =
        std::make_shared<DelayedGraphBacking>(std::as_bytes(std::span{values}));
    auto delayed_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto delayed_input =
        virtual_buffer<std::uint64_t>(ReduceElements, delayed_input_backing);
    auto delayed_output =
        virtual_buffer<std::uint64_t>(1u, delayed_output_backing);
    auto delayed_graph =
        delayed_input && delayed_output
            ? virtual_pipeline(*mapped_reduce_program, *delayed_input,
                               *delayed_output, ResidencyConfig{})
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    const Status delayed_status = delayed_graph
                                      ? delayed_graph->run()
                                      : Status::fail(delayed_graph.reason());
    const Stats delayed_stats =
        delayed_graph ? delayed_graph->stats() : Stats{};
    const ResidencyStats &delayed_residency = delayed_stats.pipeline.residency;
    const bool delayed_device_vsm =
        backend != Backend::Cpu &&
        delayed_residency.window_handoff_count == 1u &&
        delayed_residency.window_batch_count == 1u &&
        delayed_residency.window_queue_call_count == 1u;
    const bool delayed_io_exact =
        delayed_device_vsm
            ? delayed_input_backing->max_active_reads() == 1u &&
                  delayed_residency.late_page_count == 0u &&
                  delayed_residency.prefetch_count == 0u &&
                  delayed_residency.stall_ns == 0u
            : delayed_input_backing->max_active_reads() ==
                      (backend == Backend::Cpu ? 1u : 2u) &&
                  delayed_residency.late_page_count ==
                      (backend == Backend::Cpu ? ReducePages : 2u) &&
                  delayed_residency.prefetch_count ==
                      (backend == Backend::Cpu ? 0u : ReducePages - 2u) &&
                  delayed_residency.stall_ns != 0u;
    if (!delayed_status || !observe_u64(*delayed_output_backing, observed) ||
        observed != expected_mapped_sum || !delayed_io_exact ||
        delayed_residency.page_in_count != ReducePages ||
        delayed_residency.backing_read_bytes != sizeof(values)) {
      std::fprintf(
          stderr,
          "virtual mapped reduce persistent backend=%u reason=%.*s reads=%u "
          "loads=%llu late=%llu prefetch=%llu backing=%llu stall=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<int>(delayed_status.error().size()),
          delayed_status.error().data(),
          delayed_input_backing->max_active_reads(),
          static_cast<unsigned long long>(delayed_residency.page_in_count),
          static_cast<unsigned long long>(delayed_residency.late_page_count),
          static_cast<unsigned long long>(delayed_residency.prefetch_count),
          static_cast<unsigned long long>(delayed_residency.backing_read_bytes),
          static_cast<unsigned long long>(delayed_residency.stall_ns));
      std::fprintf(stderr, "virtual reduce branch=delayed-persistent\n");
      return 12;
    }
  }

  return 0;
}

} // namespace rund_node_test_virtual::product::reduce
