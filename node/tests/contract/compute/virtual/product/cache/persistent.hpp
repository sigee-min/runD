#pragma once

#include "../backing.hpp"
#include "../golden.hpp"
#include "../model.hpp"
#include "../route.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product::cache {

using StagedLoopOwner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using StagedLoopEvidence = rund::compute::detail::device_vsm_product_detail::
    DeviceVsmProductEvidence;

[[nodiscard]] inline bool StagedLoopRoute(
    const rund::compute::Status &run,
    const ProductRouteObservation &route) noexcept {
  return run && route.demand == RouteDemand::Natural &&
         route.kind == RouteKind::DeviceVsm && route.completed &&
         route.accepted_owner_mask == OwnerDeviceVsm &&
         route.accepted_owner_count == 1u && route.conflict_count == 0u &&
         route.device_vsm_prepare_called &&
         static_cast<bool>(route.device_vsm_prepare_status) &&
         route.device_vsm_execute_called &&
         route.device_vsm_execute_accepted &&
         !route.sliding_prepare_called && !route.sliding_execute_called &&
         !route.sliding_execute_accepted &&
         !route.submit_residency_pipeline_called &&
         !route.submit_residency_pipeline_accepted &&
         !route.submit_residency_stream_window_called &&
         !route.submit_residency_stream_window_accepted &&
         !route.submit_residency_schedule_called &&
         !route.submit_residency_schedule_accepted &&
         !route.prepare_residency_sliding_called &&
         !route.submit_residency_sliding_called &&
         !route.submit_residency_sliding_accepted &&
         !route.window_submit_called && !route.window_submit_accepted &&
         !route.residency_submit_failed &&
         !route.residency_submit_owner_valid;
}

template <typename Pipeline>
[[nodiscard]] inline std::shared_ptr<StagedLoopOwner>
StagedLoopOwnerOf(Pipeline &pipeline) noexcept {
  const auto &state =
      rund::compute::detail::VirtualPipelineAccess::state(pipeline);
  return state == nullptr
             ? std::shared_ptr<StagedLoopOwner>{}
             : std::static_pointer_cast<StagedLoopOwner>(
                   state->device_vsm_product_cache);
}

[[nodiscard]] inline bool StagedLoopOwnerEvidence(
    const StagedLoopOwner *const owner, const StagedLoopEvidence &evidence,
    const std::uint64_t cold_prepare, const std::uint64_t warm_rearm) noexcept {
  if (owner == nullptr ||
      owner->route_proof.kind !=
          rund::compute::detail::VirtualDeviceVsmRouteKind::StagedLoop) {
    return false;
  }
  const auto &native = evidence.native;
  return evidence.cold_prepare_count == cold_prepare &&
         evidence.warm_rearm_count == warm_rearm &&
         evidence.public_handoff_count == 1u &&
         evidence.authority_accept_count == 1u &&
         evidence.pipeline_terminal_count == 2u &&
         evidence.backing_publication_count == 1u && evidence.final_received &&
         !evidence.quarantined &&
         evidence.public_resident_input_count == 0u &&
         evidence.whole_run_staged_input_count == 1u &&
         !evidence.public_resident_output &&
         evidence.whole_run_staged_output &&
         !evidence.bounded_external_page_service &&
         evidence.final_terminal ==
             rund::node::accel::detail::DeviceVsmTerminal::Known &&
         native.page_count == PageCount &&
         native.generated_epochs == PageCount &&
         native.completed_epochs == PageCount &&
         native.forecasted_pages == PageCount &&
         native.promoted_pages == PageCount &&
         native.drained_pages == PageCount &&
         native.persisted_pages == PageCount &&
         native.gpu_backing_read_bytes == LogicalBytes &&
         native.gpu_backing_write_bytes == LogicalBytes &&
         native.native_submit_count == 1u &&
         native.epoch_native_submit_count == 0u &&
         native.payload_dispatch_count == 1u &&
         native.host_service_turn_count == 0u &&
         native.host_epoch_callback_count == 0u &&
         native.final_callback_count == 1u &&
         native.max_live_frames == FrameCapacity && native.may_write;
}

[[nodiscard]] inline bool
StagedLoopStats(const rund::compute::Stats &snapshot) noexcept {
  const auto &stats = snapshot.pipeline.residency;
  return snapshot.command_submits == 1u && snapshot.dispatches == 1u &&
         snapshot.final_dispatches == 1u &&
         snapshot.command_inflight_peak == 1u &&
         snapshot.uploaded_bytes == 0u && snapshot.downloaded_bytes == 0u &&
         snapshot.transfer_submissions.host_to_device == 0u &&
         snapshot.transfer_submissions.device_to_host == 0u &&
         snapshot.transfer_submissions.device_to_device == 0u &&
         stats.logical_bytes == 2u * LogicalBytes &&
         stats.page_bytes == ResidencyPageBytes &&
         stats.page_count == PageCount &&
         stats.frame_capacity == FrameCapacity &&
         stats.resident_frames_peak == 2u * FrameCapacity &&
         stats.epoch_count == EpochCount &&
         stats.window_handoff_count == 1u &&
         stats.window_batch_count == 1u &&
         stats.window_queue_call_count == 1u &&
         stats.page_in_count == PageCount && stats.page_out_count == PageCount &&
         stats.backing_read_bytes == LogicalBytes &&
         stats.backing_write_bytes == LogicalBytes &&
         stats.page_in_bytes == LogicalBytes &&
         stats.page_out_bytes == LogicalBytes && stats.cache_hit_count == 0u &&
         stats.eviction_count == 0u && stats.late_page_count == 0u &&
         stats.prefetch_count == 0u &&
         stats.failed_page == rund::compute::ResidencyStats::no_failed_page;
}

[[nodiscard]] inline bool StagedLoopFacts(
    const BackingFacts &input_before, const BackingFacts &input_after,
    const BackingFacts &output_before, const BackingFacts &output_after,
    const std::uint64_t version_before,
    const std::uint64_t version_after) noexcept {
  return input_after.read_count == input_before.read_count + PageCount &&
         input_after.read_bytes == input_before.read_bytes + LogicalBytes &&
         input_after.read_failure_count == input_before.read_failure_count &&
         output_after.write_count == output_before.write_count + PageCount &&
         output_after.write_bytes == output_before.write_bytes + LogicalBytes &&
         output_after.write_failure_count == output_before.write_failure_count &&
         output_after.partial_write_bytes == output_before.partial_write_bytes &&
         version_after == version_before + 1u;
}

struct StagedLoopFailure final {
  const ProductRouteObservation &route;
  const rund::compute::Status &warm;
  const BackingFacts &input_before;
  const BackingFacts &input_after;
  const BackingFacts &output_before;
  const BackingFacts &output_after;
  rund::compute::Stats run_stats{};
  rund::compute::ResidencyStats stats{};
  rund::compute::MemoryStats memory{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
};

inline void ReportStagedLoopFailure(
    const StagedLoopFailure &failure) noexcept {
  const auto &route = failure.route;
  const auto &warm = failure.warm;
  const auto &stats = failure.stats;
  std::fprintf(
      stderr,
      "virtual host tier kind=%u status=%.*s reads=%llu->%llu "
      "in=%llu hits=%llu backing=%llu late=%llu prefetch=%llu "
      "handoff=%llu batch=%llu queue=%llu epochs=%llu commands=%llu "
      "owners=0x%08x/%u conflicts=%u version=%llu->%llu "
      "writes=%llu/%llu bytes=%llu/%llu\n",
      static_cast<unsigned>(route.kind), static_cast<int>(warm.error().size()),
      warm.error().data(),
      static_cast<unsigned long long>(failure.input_before.read_count),
      static_cast<unsigned long long>(failure.input_after.read_count),
      static_cast<unsigned long long>(stats.page_in_count),
      static_cast<unsigned long long>(stats.cache_hit_count),
      static_cast<unsigned long long>(stats.backing_read_bytes),
      static_cast<unsigned long long>(stats.late_page_count),
      static_cast<unsigned long long>(stats.prefetch_count),
      static_cast<unsigned long long>(stats.window_handoff_count),
      static_cast<unsigned long long>(stats.window_batch_count),
      static_cast<unsigned long long>(stats.window_queue_call_count),
      static_cast<unsigned long long>(stats.epoch_count),
      static_cast<unsigned long long>(failure.run_stats.command_submits),
      route.accepted_owner_mask, route.accepted_owner_count,
      route.conflict_count,
      static_cast<unsigned long long>(failure.version_before),
      static_cast<unsigned long long>(failure.version_after),
      static_cast<unsigned long long>(failure.output_before.write_count),
      static_cast<unsigned long long>(failure.output_after.write_count),
      static_cast<unsigned long long>(failure.output_before.write_bytes),
      static_cast<unsigned long long>(failure.output_after.write_bytes));
  std::fprintf(
      stderr,
      "virtual host sliding prepare=%u/%.*s execute=%u/%.*s accepted=%u "
      "residency_prepare=%u/%.*s residency_submit=%u/%.*s accepted=%u\n",
      route.sliding_prepare_called ? 1u : 0u,
      static_cast<int>(route.sliding_prepare_status.error().size()),
      route.sliding_prepare_status.error().data(),
      route.sliding_execute_called ? 1u : 0u,
      static_cast<int>(route.sliding_execute_status.error().size()),
      route.sliding_execute_status.error().data(),
      route.sliding_execute_accepted ? 1u : 0u,
      route.prepare_residency_sliding_called ? 1u : 0u,
      static_cast<int>(route.prepare_residency_sliding_status.error().size()),
      route.prepare_residency_sliding_status.error().data(),
      route.submit_residency_sliding_called ? 1u : 0u,
      static_cast<int>(route.submit_residency_sliding_status.error().size()),
      route.submit_residency_sliding_status.error().data(),
      route.submit_residency_sliding_accepted ? 1u : 0u);
  std::fprintf(stderr,
               "virtual host pipeline_submit called=%u accepted=%u failed=%u "
               "reason=%.*s owner_valid=%u owner_ok=%u\n",
               route.submit_residency_pipeline_called ? 1u : 0u,
               route.submit_residency_pipeline_accepted ? 1u : 0u,
               route.residency_submit_failed ? 1u : 0u,
               static_cast<int>(route.residency_submit_reason_length),
               route.residency_submit_reason.data(),
               route.residency_submit_owner_valid ? 1u : 0u,
               route.residency_submit_owner.ok ? 1u : 0u);
}

template <typename Opened, typename Program>
[[nodiscard]] inline int CheckStagedLoop(const rund::compute::Backend backend,
                                         Opened &opened, Program &program) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto input_store =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_store =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  std::array<std::int32_t, LogicalElements> tier_seed{};
  SeedInput(tier_seed);
  if (!input_store->seed(std::as_bytes(std::span{tier_seed}))) {
    return 18;
  }
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_store);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_store);
  // StagedLoop owns two device frame banks.
  constexpr std::size_t StagedLoopBytes = FrameBytes * 2u;
  const ResidencyConfig config{
      .device_resident_bytes = StagedLoopBytes,
      .host_resident_bytes = ResidencyPageBytes * 6u,
  };
  auto pipeline =
      input && output
          ? virtual_pipeline(*program, *input, *output, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!pipeline) {
    return 18;
  }

  using BackingAccess = detail::VirtualBackingAccess;
  const BackingFacts cold_input_before = input_store->facts();
  const BackingFacts cold_output_before = output_store->facts();
  const std::uint64_t cold_version_before =
      BackingAccess::version(*output_store);
  ProductRouteObservation cold_route{};
  Status cold = Status::fail(Reason::PipelineInvalid);
  {
    ProductRouteScope scope{*opened, cold_route, RouteDemand::Natural};
    if (!scope) {
      return 19;
    }
    cold = pipeline->run();
  }
  ResolveProductRoute(cold_route, backend, static_cast<bool>(cold));
  const Stats cold_stats = pipeline->stats();
  const BackingFacts cold_input_after = input_store->facts();
  const BackingFacts cold_output_after = output_store->facts();
  const std::uint64_t cold_version_after =
      BackingAccess::version(*output_store);
  std::array<std::int32_t, LogicalElements> cold_output{};
  const bool cold_output_valid =
      output_store->observe(std::as_writable_bytes(std::span{cold_output})) &&
      output_store->tail_poisoned() &&
      GoldenMatches(std::span<const std::int32_t>{cold_output});
  StagedLoopEvidence cold_evidence{};
  const auto cold_owner = StagedLoopOwnerOf(*pipeline);
  if (cold_owner != nullptr && cold_owner->evidence != nullptr) {
    cold_evidence = *cold_owner->evidence;
  }
  if (!StagedLoopRoute(cold, cold_route) ||
      !StagedLoopStats(cold_stats) ||
      !StagedLoopFacts(cold_input_before, cold_input_after,
                       cold_output_before, cold_output_after,
                       cold_version_before, cold_version_after) ||
      cold_input_before.read_count != 0u ||
      cold_input_before.read_bytes != 0u ||
      cold_output_before.write_count != 0u ||
      cold_output_before.write_bytes != 0u || cold_version_before != 1u ||
      cold_version_after != 2u || !cold_output_valid ||
      !StagedLoopOwnerEvidence(cold_owner.get(), cold_evidence, 1u, 0u)) {
    ReportStagedLoopFailure(StagedLoopFailure{
        cold_route, cold, cold_input_before, cold_input_after,
        cold_output_before, cold_output_after, cold_stats,
        cold_stats.pipeline.residency, pipeline->memory(), cold_version_before,
        cold_version_after});
    return 20;
  }

  const BackingFacts warm_input_before = input_store->facts();
  const BackingFacts warm_output_before = output_store->facts();
  const std::uint64_t warm_version_before =
      BackingAccess::version(*output_store);
  ProductRouteObservation warm_route{};
  Status warm = Status::fail(Reason::PipelineInvalid);
  {
    ProductRouteScope scope{*opened, warm_route, RouteDemand::Natural};
    if (!scope) {
      return 21;
    }
    warm = pipeline->run();
  }
  ResolveProductRoute(warm_route, backend, static_cast<bool>(warm));
  const Stats warm_stats = pipeline->stats();
  const BackingFacts warm_input_after = input_store->facts();
  const BackingFacts warm_output_after = output_store->facts();
  const std::uint64_t warm_version_after =
      BackingAccess::version(*output_store);
  std::array<std::int32_t, LogicalElements> warm_output{};
  const bool warm_output_valid =
      output_store->observe(std::as_writable_bytes(std::span{warm_output})) &&
      output_store->tail_poisoned() &&
      GoldenMatches(std::span<const std::int32_t>{warm_output});
  StagedLoopEvidence warm_evidence{};
  const auto warm_owner = StagedLoopOwnerOf(*pipeline);
  if (warm_owner != nullptr && warm_owner->evidence != nullptr) {
    warm_evidence = *warm_owner->evidence;
  }
  const bool warm_facts_exact =
      warm_input_before.read_count == PageCount &&
      warm_input_before.read_bytes == LogicalBytes &&
      warm_output_before.write_count == PageCount &&
      warm_output_before.write_bytes == LogicalBytes &&
      warm_version_before == 2u &&
      warm_input_after.read_count == 2u * PageCount &&
      warm_input_after.read_bytes == 2u * LogicalBytes &&
      warm_output_after.write_count == 2u * PageCount &&
      warm_output_after.write_bytes == 2u * LogicalBytes &&
      warm_version_after == 3u;
  const bool warm_ok =
      StagedLoopRoute(warm, warm_route) &&
      StagedLoopStats(warm_stats) &&
      StagedLoopFacts(warm_input_before, warm_input_after,
                      warm_output_before, warm_output_after,
                      warm_version_before, warm_version_after) &&
      warm_facts_exact && warm_output_valid &&
      StagedLoopOwnerEvidence(warm_owner.get(), warm_evidence, 1u, 1u);
  if (!warm_ok) {
    ReportStagedLoopFailure(StagedLoopFailure{
        warm_route, warm, warm_input_before, warm_input_after,
        warm_output_before, warm_output_after, warm_stats,
        warm_stats.pipeline.residency, pipeline->memory(), warm_version_before,
        warm_version_after});
    return 22;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::cache
