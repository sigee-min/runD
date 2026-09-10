#include "local.hpp"
#include "sequence/local.hpp"

namespace rund_node_test_virtual::product::reduce {
namespace {

template <std::size_t Depth, std::uint64_t First, class Expression>
[[nodiscard]] constexpr auto balanced_expression(Expression value) {
  if constexpr (Depth == 0u) {
    return value ^ First;
  } else {
    constexpr std::uint64_t Width = std::uint64_t{1u} << (Depth - 1u);
    return balanced_expression<Depth - 1u, First>(value) ^
           balanced_expression<Depth - 1u, First + Width>(value);
  }
}

template <std::size_t Index> struct DeepField final {};

[[nodiscard]] constexpr std::uint64_t
balanced_value(const std::uint64_t value, const std::uint64_t first,
               const std::uint64_t depth) noexcept {
  if (depth == 0u) {
    return value ^ first;
  }
  const std::uint64_t width = std::uint64_t{1u} << (depth - 1u);
  return balanced_value(value, first, depth - 1u) ^
         balanced_value(value, first + width, depth - 1u);
}

} // namespace

int RunSequenceReduce(ReduceFixture &fixture,
                      const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (!fixture.device || !fixture.input) {
    return 12;
  }
  auto *device = &*fixture.device;
  auto *input = &*fixture.input;
  const auto &values = fixture.values;
  std::uint64_t observed = 0u;
  // Each authored Map remains independently below the 1024-node kernel
  // envelope. Private Host GraphStageSequence keeps both Maps in one
  // compiled prefix Program, so
  // The tiled shape is the fixed sequence contract: 1R->2W, then 2R->1W.
  // Its two Map stages exercise the explicit intermediate alias proof.
  {
    DeviceVsmBypassScope lending_route{*device};
    if (!lending_route) {
      std::fprintf(stderr, "virtual reduce branch=tiled-bypass-scope\n");
      return 12;
    }
    auto tiled_program =
        on(*device)
            .input<std::uint64_t>(ReduceFrameElements)
            .map("virtual-tiled-deep-first",
                 [](auto value) {
                   return record(
                       field<DeepField<0u>>(balanced_expression<7u, 1u>(value)),
                       field<DeepField<1u>>(
                           balanced_expression<7u, 129u>(value)));
                 })
            .branch([](auto fields) {
              return zip(fields.template get<DeepField<0u>>(),
                         fields.template get<DeepField<1u>>())
                  .map("virtual-tiled-deep-second",
                       [](auto left, auto right) {
                         return balanced_expression<7u, 257u>(left) ^
                                balanced_expression<7u, 385u>(right);
                       })
                  .reduce(Reduce::Sum);
            })
            .compile();
    auto tiled_slices =
        tiled_program ? detail::graph_compile::compile_tiled_graph_slices(
                            detail::FlowAccess::state(*tiled_program))
                      : Result<detail::graph_compile::TiledGraphSlices>::fail(
                            Reason::PipelineInvalid);
    auto tiled_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(std::uint64_t), sizeof(std::uint64_t));
    auto tiled_output = virtual_buffer<std::uint64_t>(1u, tiled_output_backing);
    auto tiled =
        tiled_program && tiled_output
            ? virtual_pipeline(*tiled_program, *input, *tiled_output,
                               ResidencyConfig{})
            : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                  Reason::PipelineInvalid);
    ProductRouteObservation tiled_route_observation{};
    ProductRouteScope tiled_route_scope{*device, tiled_route_observation};
    if (!tiled_route_scope) {
      std::fprintf(stderr, "virtual reduce branch=tiled-route-scope\n");
      return 12;
    }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
    const auto tiled_state =
        tiled ? detail::VirtualPipelineAccess::state(*tiled) : nullptr;
    const bool tiled_sequence_before =
        backend != Backend::Vulkan ||
        sequence_detail::InspectGraphSequence(tiled_state, false);
#else
    const bool tiled_sequence_before = true;
#endif
    constexpr std::uint64_t TiledActive = ActiveMapped;
    std::uint64_t expected_tiled = 0u;
    for (std::size_t index = 0u; index < TiledActive; ++index) {
      const std::uint64_t left = balanced_value(values[index], 1u, 7u);
      const std::uint64_t right = balanced_value(values[index], 129u, 7u);
      expected_tiled +=
          balanced_value(left, 257u, 7u) ^ balanced_value(right, 385u, 7u);
    }
    const std::uint64_t tiled_version_before =
        detail::VirtualBackingAccess::version(*tiled_output_backing);
    Status tiled_status =
        tiled ? tiled->run(TiledActive) : Status::fail(tiled.reason());
    const PipelinePlan tiled_plan = tiled ? tiled->plan() : PipelinePlan{};
    const MemoryStats tiled_memory = tiled ? tiled->memory() : MemoryStats{};
    if (tiled && !tiled->begin_samples()) {
      std::fprintf(stderr,
                   "virtual reduce branch=tiled-begin-samples status=%.*s\n",
                   static_cast<int>(tiled_status.error().size()),
                   tiled_status.error().data());
      return 12;
    }
    node_compute_allocation::Start();
    std::size_t tiled_warm_attempts = 0u;
    for (std::size_t attempt = 0u; attempt < GraphWarmRuns && tiled_status;
         ++attempt) {
      tiled_status = tiled->run(TiledActive);
      tiled_warm_attempts = attempt + 1u;
    }
    node_compute_allocation::Stop();
    const auto failed_state =
        tiled ? detail::VirtualPipelineAccess::state(*tiled) : nullptr;
    const Status tiled_end_status =
        tiled ? tiled->end_samples() : Status::success();
    if (tiled && !tiled_end_status) {
      const ResidencyStats tiled_end_residency =
          tiled->stats().pipeline.residency;
      std::fprintf(stderr,
                   "virtual reduce branch=tiled-end-samples "
                   "end_status=%u end_reason=%u/%.*s "
                   "run_status=%u run_reason=%u/%.*s "
                   "sampled_runs=%u allocation_free_runs=%u\n",
                   tiled_end_status ? 1u : 0u,
                   static_cast<unsigned>(tiled_end_status.reason()),
                   static_cast<int>(tiled_end_status.error().size()),
                   tiled_end_status.error().data(), tiled_status ? 1u : 0u,
                   static_cast<unsigned>(tiled_status.reason()),
                   static_cast<int>(tiled_status.error().size()),
                   tiled_status.error().data(),
                   tiled_end_residency.sampled_runs,
                   tiled_end_residency.allocation_free_runs);
      const bool tiled_valid = detail::valid_virtual_pipeline(failed_state);
      std::fprintf(stderr,
                   "virtual reduce branch=tiled-end-state phase=%u "
                   "samples=%u valid_virtual_pipeline=%u "
                   "warm_attempts=%zu\n",
                   failed_state == nullptr
                       ? 255u
                       : static_cast<unsigned>(failed_state->phase),
                   failed_state == nullptr
                       ? 255u
                       : static_cast<unsigned>(failed_state->samples),
                   tiled_valid ? 1u : 0u, tiled_warm_attempts);
      if (failed_state != nullptr && failed_state->failure_log.has()) {
        const auto &failure = failed_state->failure_log.first();
        std::fprintf(
            stderr,
            "virtual reduce branch=tiled-end-failure-log phase=%u check=%u "
            "reason=%u stage=%u batch=%llu epoch=%llu token=%llu "
            "generation=%llu\n",
            static_cast<unsigned>(failure.phase),
            static_cast<unsigned>(failure.check),
            static_cast<unsigned>(failure.reason),
            static_cast<unsigned>(failure.stage),
            static_cast<unsigned long long>(failure.batch),
            static_cast<unsigned long long>(failure.epoch),
            static_cast<unsigned long long>(failure.token),
            static_cast<unsigned long long>(failure.generation));
        if (failure.has_close) {
          const auto &close = failure.close;
          std::fprintf(
              stderr,
              "virtual reduce branch=tiled-end-failure-close "
              "check=%u token=%llu generation=%llu frame=%u binding=%u "
              "claims=%u extent=%llu view=%llu "
              "bkey=%llu/%llu/%llu/%llu/%llu/%llu/%u "
              "fkey=%llu/%llu/%llu/%llu/%llu/%llu/%u "
              "baccess=%u bflags=%u bfetch=%u brelocated=%u bretire=%u "
              "bnext=%llu bretain=%llu fstate=%u ftier=%u frole=%u "
              "fassigned=%u fnext=%llu fretain=%llu bdirty=%llu/%llu "
              "fdirty=%llu/%llu prior=%llu/%llu\n",
              static_cast<unsigned>(close.check),
              static_cast<unsigned long long>(close.token),
              static_cast<unsigned long long>(close.generation),
              close.frame_index, close.binding_index, close.claims,
              static_cast<unsigned long long>(close.extent),
              static_cast<unsigned long long>(close.view),
              static_cast<unsigned long long>(close.binding_key.backing),
              static_cast<unsigned long long>(close.binding_key.version),
              static_cast<unsigned long long>(close.binding_key.extent),
              static_cast<unsigned long long>(
                  close.binding_key.materialization_hi),
              static_cast<unsigned long long>(
                  close.binding_key.materialization_lo),
              static_cast<unsigned long long>(close.binding_key.page),
              static_cast<unsigned>(close.binding_key.domain),
              static_cast<unsigned long long>(close.frame_key.backing),
              static_cast<unsigned long long>(close.frame_key.version),
              static_cast<unsigned long long>(close.frame_key.extent),
              static_cast<unsigned long long>(
                  close.frame_key.materialization_hi),
              static_cast<unsigned long long>(
                  close.frame_key.materialization_lo),
              static_cast<unsigned long long>(close.frame_key.page),
              static_cast<unsigned>(close.frame_key.domain),
              static_cast<unsigned>(close.binding_access),
              static_cast<unsigned>(close.flags), close.binding_fetch ? 1u : 0u,
              close.binding_relocated ? 1u : 0u, close.binding_retire ? 1u : 0u,
              static_cast<unsigned long long>(close.binding_next_use),
              static_cast<unsigned long long>(close.binding_retain_until),
              static_cast<unsigned>(close.frame_state),
              static_cast<unsigned>(close.frame_tier),
              static_cast<unsigned>(close.frame_role),
              close.frame_assigned ? 1u : 0u,
              static_cast<unsigned long long>(close.frame_next_use),
              static_cast<unsigned long long>(close.frame_retain_until),
              static_cast<unsigned long long>(close.binding_dirty.offset),
              static_cast<unsigned long long>(close.binding_dirty.bytes),
              static_cast<unsigned long long>(close.frame_dirty.offset),
              static_cast<unsigned long long>(close.frame_dirty.bytes),
              static_cast<unsigned long long>(close.binding_prior_dirty.offset),
              static_cast<unsigned long long>(close.binding_prior_dirty.bytes));
        }
      }
      return 12;
    }
    const std::uint64_t tiled_pages =
        (TiledActive + ReduceFrameElements - 1u) / ReduceFrameElements;
    const std::uint64_t tiled_batches =
        tiled ? (tiled_pages + tiled->plan().residency.frame_capacity - 1u) /
                    tiled->plan().residency.frame_capacity
              : 0u;
    const ResidencyStats tiled_residency =
        tiled ? tiled->stats().pipeline.residency : ResidencyStats{};
    const bool tiled_output_observed =
        tiled && observe_u64(*tiled_output_backing, observed);
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
    const bool tiled_sequence_after =
        backend != Backend::Vulkan ||
        sequence_detail::InspectGraphSequence(tiled_state, true);
#else
    const bool tiled_sequence_after = true;
#endif
    const std::uint64_t tiled_version_after =
        detail::VirtualBackingAccess::version(*tiled_output_backing);
    if (!tiled_program || !tiled_slices || tiled_slices->stages.size() != 2u ||
        tiled_slices->stages[0u].inputs.size() != 1u ||
        tiled_slices->stages[0u].outputs.size() != 1u ||
        tiled_slices->stages[1u].inputs.size() != 1u ||
        tiled_slices->stages[1u].outputs.size() != 1u || !tiled ||
        !tiled_status || tiled->plan().residency.frame_capacity != 2u ||
        !tiled_output_observed || observed != expected_tiled ||
        !tiled_sequence_before || !tiled_sequence_after ||
        tiled_version_before >
            std::numeric_limits<std::uint64_t>::max() - (GraphWarmRuns + 1u) ||
        tiled_version_after != tiled_version_before + (GraphWarmRuns + 1u) ||
        tiled_residency.epoch_count != tiled_batches * 2u ||
        !allocation_boundary_exact(backend, node_compute_allocation::Count()) ||
        tiled->plan() != tiled_plan ||
        !same_fixed_memory(tiled->memory(), tiled_memory) ||
        !tiled_residency.samples_allocation_free(GraphWarmRuns) ||
        tiled->stats().buffer_allocations != 0u) {
      std::fprintf(
          stderr,
          "virtual tiled graph backend=%u reason=%.*s stages=%zu observed=%llu "
          "expected=%llu k=%llu epochs=%llu expected_epochs=%llu alloc=%llu "
          "handoff=%llu batches=%llu queues=%llu submits=%llu "
          "dispatches=%llu sequence_before=%u sequence_after=%u "
          "version=%llu->%llu\n",
          static_cast<unsigned>(backend),
          static_cast<int>(tiled_status.error().size()),
          tiled_status.error().data(),
          tiled_slices ? tiled_slices->stages.size() : 0u,
          static_cast<unsigned long long>(observed),
          static_cast<unsigned long long>(expected_tiled),
          static_cast<unsigned long long>(
              tiled ? tiled->plan().residency.frame_capacity : 0u),
          static_cast<unsigned long long>(tiled_residency.epoch_count),
          static_cast<unsigned long long>(tiled_batches * 2u),
          static_cast<unsigned long long>(node_compute_allocation::Count()),
          static_cast<unsigned long long>(tiled_residency.window_handoff_count),
          static_cast<unsigned long long>(tiled_residency.window_batch_count),
          static_cast<unsigned long long>(
              tiled_residency.window_queue_call_count),
          static_cast<unsigned long long>(tiled ? tiled->stats().command_submits
                                                : 0u),
          static_cast<unsigned long long>(tiled ? tiled->stats().dispatches
                                                : 0u),
          tiled_sequence_before ? 1u : 0u, tiled_sequence_after ? 1u : 0u,
          static_cast<unsigned long long>(tiled_version_before),
          static_cast<unsigned long long>(tiled_version_after));
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
      sequence_detail::PrintTiledGraphEvidence(
          backend,
          tiled ? detail::VirtualPipelineAccess::state(*tiled) : nullptr,
          tiled_route_observation);
#endif
      std::fprintf(stderr, "virtual reduce branch=tiled-oracle\n");
      return 12;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::reduce
