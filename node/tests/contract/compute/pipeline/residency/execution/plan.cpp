#include "local.hpp"

#include <bit>

namespace rund_node_test_pipeline_residency::execution_test {

[[nodiscard]] int CheckExecutionPlan() {
  const execution::Request request = MakeRequest();
  const execution::SealResult sealed = execution::seal(request);
  if (!sealed || sealed.plan.identity() == 0u ||
      sealed.plan.epoch_count() != 4u || !sealed.plan.external_all_or_none()) {
    return 1;
  }
  execution::Forecast forecast0{};
  execution::Forecast forecast3{};
  execution::Request different_horizon = request;
  different_horizon.prefetch_distance = 3u;
  const execution::SealResult horizon = execution::seal(different_horizon);
  if (!sealed.plan.forecast(0u, forecast0) ||
      !sealed.plan.forecast(3u, forecast3) ||
      forecast0 != execution::Forecast{.epoch = 0u,
                                       .first_page = 0u,
                                       .page_count = 3u,
                                       .prefetch_epoch = 0u,
                                       .ready_epoch = 0u} ||
      forecast3 != execution::Forecast{.epoch = 3u,
                                       .first_page = 9u,
                                       .page_count = 1u,
                                       .prefetch_epoch = 1u,
                                       .ready_epoch = 3u} ||
      sealed.plan.forecast(4u, forecast3) || !horizon ||
      horizon.plan.identity() == sealed.plan.identity()) {
    return 1;
  }
  execution::Request host_ring_request = request;
  host_ring_request.host_input[0].count = 5u;
  host_ring_request.host_input[1].first = 5u;
  host_ring_request.host_input[1].count = 5u;
  host_ring_request.host_output[0].count = 5u;
  host_ring_request.host_output[1].first = 23u;
  host_ring_request.host_output[1].count = 5u;
  const execution::SealResult host_ring = execution::seal(host_ring_request);
  execution::Node host_ring_input{};
  if (!host_ring || host_ring.plan.epoch_count() != 4u ||
      !host_ring.plan.project(
          execution::NodeId{.epoch = 1u, .phase = execution::Phase::Input},
          host_ring_input) ||
      host_ring_input.input_count != 3u ||
      host_ring_input.route.source != host_ring_request.host_input[1] ||
      host_ring_input.route.target != host_ring_request.device_input[1] ||
      host_ring.plan.host_output_regions() != host_ring_request.host_output) {
    return 2;
  }
  execution::Request asymmetric_host_ring = host_ring_request;
  asymmetric_host_ring.host_input[1].count = 4u;
  if (execution::seal(asymmetric_host_ring)) {
    return 3;
  }
  asymmetric_host_ring = host_ring_request;
  asymmetric_host_ring.host_output[1].count = 4u;
  if (execution::seal(asymmetric_host_ring)) {
    return 3;
  }
  execution::Request transient_input = request;
  transient_input.input.cache.key.domain = residency::CacheDomain::Transient;
  execution::Request transient_output = request;
  transient_output.output.cache.key.domain = residency::CacheDomain::Transient;
  execution::Request overflowing_next_use = request;
  overflowing_next_use.input.next_use_base =
      std::numeric_limits<std::uint64_t>::max() - 1u;
  overflowing_next_use.input.next_use_stride = 2u;
  execution::Request past_next_use = request;
  past_next_use.input.next_use_base = 1u;
  past_next_use.input.next_use_stride = 0u;
  if (execution::seal(transient_input) || execution::seal(transient_output) ||
      execution::seal(overflowing_next_use) || execution::seal(past_next_use)) {
    return 3;
  }

  execution::Request zero_tail_request = request;
  zero_tail_request.input.fill = execution::FetchFill::ZeroInactiveTail;
  const execution::SealResult zero_tail = execution::seal(zero_tail_request);
  execution::FetchSource zero_tail_source{};
  if (sealed.plan.input_sources_materializable() || !zero_tail ||
      zero_tail.plan.identity() == sealed.plan.identity() ||
      !zero_tail.plan.input_sources_materializable() ||
      !zero_tail.plan.input_source(9u, zero_tail_source) ||
      zero_tail_source.complete_frame() ||
      !zero_tail_source.materializes_frame() ||
      zero_tail_source.fill != execution::FetchFill::ZeroInactiveTail ||
      zero_tail_source.bytes != 64u || zero_tail_source.frame_bytes != 128u) {
    return 3;
  }

  execution::Request halo_request = request;
  halo_request.input.cache.key.materialization_hi = 0x48414c4f5f494e50ull;
  halo_request.input.cache.key.materialization_lo = 0x55545f4c41594f55ull;
  halo_request.input.cache.page_bytes = 192u;
  halo_request.input.read_prefix_bytes = 32u;
  halo_request.input.target_prefix_bytes = 32u;
  halo_request.input.read_suffix_bytes = 32u;
  execution::Request invalid_zero_halo = halo_request;
  invalid_zero_halo.input.fill = execution::FetchFill::ZeroInactiveTail;
  const execution::SealResult halo = execution::seal(halo_request);
  execution::FetchSource halo_first{};
  execution::FetchSource halo_middle{};
  execution::FetchSource halo_tail{};
  if (execution::seal(invalid_zero_halo) || !halo ||
      !halo.plan.input_source(0u, halo_first) ||
      !halo.plan.input_source(1u, halo_middle) ||
      !halo.plan.input_source(9u, halo_tail) ||
      halo_first != execution::FetchSource{.key = halo_first.key,
                                           .offset = 0u,
                                           .bytes = 160u,
                                           .target_offset = 32u,
                                           .frame_bytes = 192u} ||
      halo_middle != execution::FetchSource{.key = halo_middle.key,
                                            .offset = 96u,
                                            .bytes = 192u,
                                            .target_offset = 0u,
                                            .frame_bytes = 192u} ||
      halo_tail != execution::FetchSource{.key = halo_tail.key,
                                          .offset = 1120u,
                                          .bytes = 96u,
                                          .target_offset = 0u,
                                          .frame_bytes = 192u} ||
      halo_first.key.materialization_hi !=
          halo_request.input.cache.key.materialization_hi ||
      halo_first.key.materialization_lo !=
          halo_request.input.cache.key.materialization_lo ||
      halo_first.key.page != 0u || halo_middle.key.page != 1u ||
      halo_tail.key.page != 9u) {
    return 3;
  }

  const execution::Request footprint_request = MakeFootprintRequest();
  const execution::SealResult footprint = execution::seal(footprint_request);
  execution::WindowFootprintProjection footprint0{};
  execution::WindowFootprintProjection footprint1{};
  execution::WindowFootprintProjection footprint2{};
  execution::WindowFootprintProjection invalid_footprint{};
  execution::FetchSource canonical_tail{};
  if (!footprint || !footprint.plan.has_window_footprint() ||
      !footprint.plan.window_footprint(0u, footprint0) ||
      !footprint.plan.window_footprint(1u, footprint1) ||
      !footprint.plan.window_footprint(2u, footprint2) ||
      footprint.plan.window_footprint(3u, invalid_footprint) ||
      !footprint.plan.canonical_input_source(4u, canonical_tail)) {
    return 31;
  }
  if (footprint0.epoch != 0u || footprint0.source_count != 3u ||
      footprint0.target_count != 2u || footprint0.slice_count != 5u ||
      footprint0.sources[0].key.page != 0u ||
      footprint0.sources[0].next_use != residency::NeverUse ||
      footprint0.sources[1].key.page != 1u ||
      footprint0.sources[1].next_use != 1u ||
      footprint0.sources[2].key.page != 2u ||
      footprint0.sources[2].next_use != 1u ||
      footprint0.sources[1].retain_until != 0u ||
      footprint0.targets[0].key.page != 0u ||
      footprint0.targets[1].key.page != 1u) {
    return 32;
  }
  if (footprint0.slices[0] !=
          execution::WindowFootprintSlice{.source = 0u,
                                          .target = 0u,
                                          .source_offset = 0u,
                                          .target_offset = 8u,
                                          .bytes = 48u} ||
      footprint0.slices[1] !=
          execution::WindowFootprintSlice{.source = 1u,
                                          .target = 0u,
                                          .source_offset = 0u,
                                          .target_offset = 56u,
                                          .bytes = 8u} ||
      footprint0.slices[2] !=
          execution::WindowFootprintSlice{.source = 0u,
                                          .target = 1u,
                                          .source_offset = 40u,
                                          .target_offset = 0u,
                                          .bytes = 8u} ||
      footprint0.slices[3] !=
          execution::WindowFootprintSlice{.source = 1u,
                                          .target = 1u,
                                          .source_offset = 0u,
                                          .target_offset = 8u,
                                          .bytes = 48u} ||
      footprint0.slices[4] !=
          execution::WindowFootprintSlice{.source = 2u,
                                          .target = 1u,
                                          .source_offset = 0u,
                                          .target_offset = 56u,
                                          .bytes = 8u}) {
    return 33;
  }
  if (footprint1.source_count != 4u || footprint1.target_count != 2u ||
      footprint1.slice_count != 6u || footprint1.sources[0].key.page != 1u ||
      footprint1.sources[3].key.page != 4u ||
      footprint1.sources[0].next_use != residency::NeverUse ||
      footprint1.sources[2].next_use != 2u) {
    return 34;
  }
  if (footprint2.source_count != 2u || footprint2.target_count != 1u ||
      footprint2.slice_count != 2u ||
      footprint2.slices[0].source_offset != 40u ||
      footprint2.slices[0].target_offset != 0u ||
      footprint2.slices[0].bytes != 8u ||
      footprint2.slices[1].source_offset != 0u ||
      footprint2.slices[1].target_offset != 8u ||
      footprint2.slices[1].bytes != 36u || canonical_tail.offset != 192u ||
      canonical_tail.bytes != 36u || canonical_tail.frame_bytes != 48u ||
      canonical_tail.fill != execution::FetchFill::ZeroInactiveTail ||
      !canonical_tail.materializes_frame()) {
    return 35;
  }
  execution::Request aliased_footprint = footprint_request;
  aliased_footprint.canonical_input.key = aliased_footprint.input.cache.key;
  execution::Request undersized_footprint = footprint_request;
  undersized_footprint.host_input[0].count = 3u;
  undersized_footprint.host_input[1].first = 3u;
  undersized_footprint.host_input[1].count = 3u;
  execution::Request changed_footprint = footprint_request;
  ++changed_footprint.canonical_input.key.materialization_lo;
  const execution::SealResult changed = execution::seal(changed_footprint);
  if (execution::seal(aliased_footprint) ||
      execution::seal(undersized_footprint) || !changed ||
      changed.plan.identity() == footprint.plan.identity()) {
    return 36;
  }
  execution::Request huge_source = request;
  huge_source.page_count = 2u;
  huge_source.input.cache.page_count = 2u;
  huge_source.input.cache.page_bytes =
      std::numeric_limits<std::uint64_t>::max() / 2u + 2u;
  huge_source.input.logical_bytes =
      std::numeric_limits<std::uint64_t>::max() - 1u;
  huge_source.input.payload_bytes =
      (std::numeric_limits<std::uint64_t>::max() - 1u) / 2u;
  huge_source.input.read_suffix_bytes = 2u;
  huge_source.output.cache.page_count = 2u;
  huge_source.output.logical_bytes = 2u;
  huge_source.output.payload_bytes = 1u;
  huge_source.publication.extent.bytes = 2u;
  const execution::SealResult huge = execution::seal(huge_source);
  execution::FetchSource huge_tail{};
  if (!huge || !huge.plan.input_source(1u, huge_tail) ||
      huge_tail.offset != huge_source.input.payload_bytes ||
      huge_tail.bytes != huge_source.input.payload_bytes ||
      huge_tail.target_offset != 0u) {
    return 3;
  }

  const auto live_rows =
      [&](const std::uint64_t pages, const std::uint64_t retain,
          const std::uint64_t bank0, const std::uint64_t bank1) {
        execution::Request live = request;
        live.page_count = pages;
        live.frame_capacity = 2u;
        live.input.cache.page_count = pages;
        live.input.logical_bytes = pages * live.input.payload_bytes;
        live.input.next_use_base = pages;
        live.input.retain_until = retain;
        live.output.cache.page_count = pages;
        live.output.logical_bytes = pages * live.output.payload_bytes;
        live.publication.extent.bytes = live.output.logical_bytes;
        for (std::size_t bank = 0u; bank < execution::BankCapacity; ++bank) {
          live.host_input[bank].count = 2u;
          live.device_input[bank].count = 2u;
          live.device_output[bank].count = 2u;
          live.host_output[bank].count = 2u;
        }
        const execution::SealResult projected = execution::seal(live);
        std::uint64_t actual0 = 0u;
        std::uint64_t actual1 = 0u;
        return projected && projected.plan.input_live_rows(0u, actual0) &&
               projected.plan.input_live_rows(1u, actual1) &&
               actual0 == bank0 && actual1 == bank1;
      };
  if (!live_rows(5u, 2u, 3u, 2u) || !live_rows(3u, 1u, 2u, 1u) ||
      !live_rows(6u, 2u, 4u, 2u) || !live_rows(1u, 0u, 1u, 0u) ||
      !live_rows(6u, residency::NeverUse, 2u, 2u)) {
    return 3;
  }

  execution::Node first{};
  execution::Node tail{};
  execution::Node drain{};
  if (!sealed.plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Input},
          first) ||
      !sealed.plan.project(
          execution::NodeId{.epoch = 3u, .phase = execution::Phase::Dispatch},
          tail) ||
      !sealed.plan.project(
          execution::NodeId{.epoch = 3u, .phase = execution::Phase::Output},
          drain) ||
      first.domain != execution::Domain::HostService || first.bank != 0u ||
      first.input_count != 3u || first.active_mask != 7u ||
      first.input[2].key.page != 2u || first.input[2].key.extent != 0u ||
      first.input[2].next_use != 12u || first.input[2].epoch != 0u ||
      !first.may_write || first.mutation_count != 2u ||
      first.mutations[0] != first.route.source ||
      first.mutations[1] != first.route.target ||
      first.route.source.tier != residency::FrameTier::Host ||
      tail.domain != execution::Domain::Native || tail.bank != 1u ||
      tail.input_count != 1u || tail.output_count != 1u ||
      tail.active_mask != 1u || !tail.may_write || tail.mutation_count != 1u ||
      tail.mutations[0] != tail.route.target ||
      tail.input[0].key.extent != 24'575u ||
      tail.output[0].key.extent != 24'575u ||
      tail.output[0].dirty !=
          residency::DirtyExtent{.offset = 4816u, .bytes = 50u} ||
      !drain.may_write || drain.domain != execution::Domain::HostService ||
      drain.mutation_count != 1u || drain.mutations[0] != drain.route.target ||
      drain.route.target.tier != residency::FrameTier::Host) {
    return 2;
  }
  if (!dependency(
          sealed.plan,
          execution::NodeId{.epoch = 2u, .phase = execution::Phase::Input},
          execution::NodeId{.epoch = 0u,
                            .phase = execution::Phase::Dispatch}) ||
      !dependency(
          sealed.plan,
          execution::NodeId{.epoch = 2u, .phase = execution::Phase::Dispatch},
          execution::NodeId{.epoch = 2u,
                            .phase = execution::Phase::Input}) ||
      !dependency(
          sealed.plan,
          execution::NodeId{.epoch = 2u, .phase = execution::Phase::Dispatch},
          execution::NodeId{.epoch = 0u,
                            .phase = execution::Phase::Output}) ||
      !dependency(
          sealed.plan,
          execution::NodeId{.epoch = 2u, .phase = execution::Phase::Output},
          execution::NodeId{.epoch = 2u,
                            .phase = execution::Phase::Dispatch})) {
    return 3;
  }

  execution::Receipt receipt{};
  if (!receipt.arm(sealed.plan, 21u, 22u, 100u) || !receipt.submit() ||
      receipt.submit() ||
      receipt.terminal(execution::Evidence{
          .status = Status::success(),
          .plan_identity = sealed.plan.identity(),
          .token = 21u,
          .generation = 23u,
          .epoch_count = 4u,
          .native_submissions = 1u,
          .progress = execution::Progress{.input_services = 4u,
                                          .native_dispatches = 4u,
                                          .native_completions = 4u,
                                          .output_services = 4u,
                                          .native_inflight_peak = 2u},
          .completed_ns = 200u}) ||
      !receipt.terminal(execution::Evidence{
          .status = Status::success(),
          .plan_identity = sealed.plan.identity(),
          .token = 21u,
          .generation = 22u,
          .epoch_count = 4u,
          .native_submissions = 1u,
          .progress = execution::Progress{.input_services = 4u,
                                          .native_dispatches = 4u,
                                          .native_completions = 4u,
                                          .output_services = 4u,
                                          .native_inflight_peak = 2u},
          .completed_ns = 200u})) {
    return 4;
  }
  const execution::ReceiptSnapshot observed = receipt.wait();
  if (!observed.status || observed.plan_identity != sealed.plan.identity() ||
      observed.token != 21u || observed.generation != 22u ||
      observed.epoch_count != 4u || observed.native_submissions != 1u ||
      observed.progress.input_services != 4u ||
      observed.progress.native_dispatches != 4u ||
      observed.progress.native_completions != 4u ||
      observed.progress.output_services != 4u ||
      observed.progress.native_inflight_peak != 2u ||
      observed.started_ns != 100u || observed.completed_ns != 200u) {
    return 5;
  }

  execution::Receipt failed{};
  if (!failed.arm(sealed.plan, 31u, 32u, 300u) || !failed.submit() ||
      !failed.terminal(execution::Evidence{
          .status = Status::fail(Reason::DeviceLost),
          .plan_identity = sealed.plan.identity(),
          .token = 31u,
          .generation = 32u,
          .epoch_count = 4u,
          .native_submissions = 1u,
          .failures = {execution::FailureEvidence{
              .epoch = 0u, .phases = 2u, .may_write = 2u, .terminal = 2u}},
          .failure_count = 1u,
          .completed_ns = 301u}) ||
      failed.wait().status.reason() != Reason::DeviceLost) {
    return 6;
  }
  if (execution::seal(execution::Request{}).failure !=
          execution::SealFailure::Invalid ||
      execution::seal(
          execution::Request{.page_count = 1u,
                             .frame_capacity = execution::UseCapacity + 1u})
              .failure != execution::SealFailure::Capacity) {
    return 7;
  }
  execution::Request wrong_publication = request;
  ++wrong_publication.publication.generation;
  wrong_publication.publication.extent.bytes = 769u;
  if (execution::seal(wrong_publication)) {
    return 8;
  }
  execution::Request partial_publication = request;
  partial_publication.publication.generation = 0u;
  partial_publication.publication.capability = 0u;
  const execution::SealResult partial = execution::seal(partial_publication);
  if (!partial || partial.plan.external_all_or_none()) {
    return 9;
  }
  execution::Request partial_capability = request;
  partial_capability.publication.capability = 0u;
  if (execution::seal(partial_capability)) {
    return 10;
  }
  execution::Request invalid_retention = request;
  invalid_retention.input.retain_until = 2u;
  if (execution::seal(invalid_retention)) {
    return 11;
  }
  execution::Request overlapping = request;
  overlapping.device_output[1].first = overlapping.device_input[1].first + 2u;
  if (execution::seal(overlapping)) {
    return 12;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::execution_test
