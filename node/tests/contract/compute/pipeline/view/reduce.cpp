#include "src/compute/pipeline/state/assembly.hpp"
#include "local.hpp"

#include "../../allocation.hpp"
#include "src/compute/pipeline/plan/arena.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>

namespace rund_node_test_pipeline::view {

[[nodiscard]] int CheckReduce(rund::compute::Device &device,
                              const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto source = Upload(device, source_values);
  if (!source) {
    return 7;
  }
  // Dense-only primitives retain the public View unchanged.  CPU performs
  // allocation-free gather/publish loops; Metal and Vulkan encode the same
  // gather/primitive/scatter sequence into the prepared device stream.  Map
  // above remains a native strided binding on every backend.
  auto reduce =
      on(device)
          .input<std::int32_t>(4u)
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  auto reduced_target = device.buffer<std::int32_t>(3u);
  auto reduce_input = source->view(1u, 4u, 2u);
  auto reduce_output = reduced_target->view(1u, 1u, 2u);
  if (!reduce || !reduced_target || !reduce_input || !reduce_output) {
    return 7;
  }
  auto reduced_builder = pipeline(device).then(*reduce, read(*reduce_input),
                                               write(*reduce_output));
  const auto reduced_plan = reduced_builder.plan();
  const std::shared_ptr<detail::DeviceState> device_state =
      detail::DeviceAccess::state(device);
  const detail::AccelDeviceState *const accel =
      device_state == nullptr ? nullptr : detail::accel_device(*device_state);
  if (accel != nullptr && !rund::kernel::ComputeStorageAlignmentValid(
                              accel->pick.caps.storage_alignment)) {
    return 8;
  }
  const std::uint64_t view_alignment =
      accel == nullptr
          ? sizeof(std::uint32_t)
          : std::max<std::uint64_t>(sizeof(std::uint32_t),
                                    accel->pick.caps.storage_alignment);
  const std::uint64_t input_bytes = 4u * sizeof(std::int32_t);
  const bool vulkan_output_dense =
      backend == Backend::Vulkan && sizeof(std::int32_t) % view_alignment != 0u;
  const std::uint64_t expected_view_buffer =
      backend == Backend::Cpu || backend == Backend::Metal ? input_bytes
      : vulkan_output_dense
          ? ((input_bytes + view_alignment - 1u) & ~(view_alignment - 1u)) +
                sizeof(std::int32_t)
          : input_bytes;
  const std::size_t expected_bindings = backend == Backend::Cpu     ? 0u
                                        : backend == Backend::Metal ? 1u
                                        : vulkan_output_dense       ? 2u
                                                                    : 1u;
  const std::size_t expected_owners = backend == Backend::Cpu ? 0u : 1u;

  // Exercise the view-summary consumer independently from nested topology.
  // Fold is reusable, so the phase is known but no physical outer coordinate
  // may be manufactured by this compact planning surface.
  if (backend != Backend::Cpu) {
    std::array<detail::PipelineBuildStep, 1u> fold_steps{};
    fold_steps.front().program = detail::ProgramAccess::state(*reduce);
    fold_steps.front().logical_step = 9u;
    fold_steps.front().iteration = 2u;
    fold_steps.front().route = detail::PipelineRoute::NestedFold;
    detail::PipelineMemoryPlan fold_plan{};
    fold_plan.job_owners = {0u};
    fold_plan.step_resources.resize(1u);
    detail::PipelineStepResourcePlan &sealed = fold_plan.step_resources.front();
    sealed.inputs.push_back(detail::PipelineResolvedViewPlan{
        .declared_backing_bytes = source_values.size() * sizeof(std::int32_t),
        .offset = 1u,
        .count = 4u,
        .stride = 2u,
        .element_bytes = sizeof(std::int32_t),
        .alignment = sizeof(std::int32_t),
        .offset_bytes = sizeof(std::int32_t),
        .stride_bytes = 2u * sizeof(std::int32_t),
        .payload_bytes = 4u * sizeof(std::int32_t),
        .span_bytes = 7u * sizeof(std::int32_t),
    });
    sealed.outputs.push_back(detail::PipelineResolvedOutputPlan{
        .view =
            detail::PipelineResolvedViewPlan{
                .declared_access = detail::ResourceAccess::Write,
                .declared_backing_bytes = 3u * sizeof(std::int32_t),
                .offset = 1u,
                .count = 1u,
                .stride = 2u,
                .element_bytes = sizeof(std::int32_t),
                .alignment = sizeof(std::int32_t),
                .offset_bytes = sizeof(std::int32_t),
                .stride_bytes = 2u * sizeof(std::int32_t),
                .payload_bytes = sizeof(std::int32_t),
                .span_bytes = sizeof(std::int32_t),
            },
        .physical = 0u,
    });
    sealed.physical_sources = {0u};
    const Status fold_planned = detail::plan_pipeline_views(
        *device_state, std::span<const detail::PipelineBuildStep>{fold_steps},
        fold_plan);
    if (!fold_planned ||
        fold_plan.summary.view_nested_phase != PipelineNestedPhase::Fold ||
        fold_plan.summary.view_step != 9u ||
        fold_plan.summary.view_iteration != 2u ||
        fold_plan.summary.view_outer_window !=
            std::numeric_limits<std::size_t>::max() ||
        fold_plan.summary.view_inner_iteration !=
            std::numeric_limits<std::size_t>::max()) {
      return 8;
    }
  }

  if (!reduced_plan ||
      reduced_plan->prepared_buffer_bytes !=
          expected_view_buffer + reduced_plan->scratch_bytes ||
      reduced_plan->prepared_bytes != reduced_plan->prepared_buffer_bytes +
                                          reduced_plan->prepared_host_bytes +
                                          reduced_plan->prepared_tile_bytes +
                                          reduced_plan->prepared_native_bytes ||
      reduced_plan->view_bytes != 4u * sizeof(std::int32_t) ||
      reduced_plan->view_span_bytes != 7u * sizeof(std::int32_t) ||
      reduced_plan->view_backing_bytes !=
          source_values.size() * sizeof(std::int32_t) ||
      reduced_plan->view_offset_bytes != sizeof(std::int32_t) ||
      reduced_plan->view_stride_bytes != 2u * sizeof(std::int32_t) ||
      reduced_plan->view_element_bytes != sizeof(std::int32_t) ||
      reduced_plan->view_count != 4u ||
      reduced_plan->view_alignment != sizeof(std::int32_t) ||
      reduced_plan->view_step != 0u || reduced_plan->view_iteration != 0u ||
      reduced_plan->view_binding == std::numeric_limits<std::size_t>::max() ||
      reduced_plan->peak_bytes != reduced_plan->state_bytes +
                                      reduced_plan->transient_bytes +
                                      reduced_plan->prepared_bytes ||
      reduced_plan->total_bytes !=
          reduced_plan->persistent_bytes + reduced_plan->peak_bytes) {
    return 8;
  }
  auto rejected =
      pipeline(device)
          .then(*reduce, read(*reduce_input), write(*reduce_output))
          .budget(MemoryBudget{.bytes = reduced_plan->peak_bytes - 1u})
          .prepare();
  if (rejected || rejected.reason() != Reason::PipelineMemoryBudget) {
    return 8;
  }
  auto reduced = std::move(reduced_builder).prepare();
  const std::shared_ptr<detail::PipelineState> reduced_state =
      reduced ? detail::PipelineStateAccess::state(*reduced)
              : std::shared_ptr<detail::PipelineState>{};
  if (reduced_state == nullptr ||
      reduced_state->prepared_buffers.size() !=
          expected_owners + reduced_plan->scratch_count ||
      reduced_state->steps.size() != 1u ||
      reduced_state->steps.front().job == nullptr ||
      (expected_owners != 0u &&
       (reduced_state->steps.front().job->workspace == nullptr ||
        reduced_state->steps.front().job->workspace->arena == nullptr ||
        reduced_state->steps.front().job->workspace->arena->binds.size() !=
            expected_bindings + reduced_plan->scratch_count)) ||
      (expected_owners == 0u &&
       reduced_state->steps.front().job->workspace != nullptr &&
       reduced_state->steps.front().job->workspace->arena != nullptr)) {
    return 8;
  }
  if ((expected_owners != 0u &&
       reduced_state->prepared_buffers.front()->bytes !=
           expected_view_buffer) ||
      (backend == Backend::Cpu &&
       (CpuViewTransferCount(*reduced_state->steps.front().job) != 1u ||
        FirstCpuViewBuffer(*reduced_state->steps.front().job) == nullptr ||
        FirstCpuViewBuffer(*reduced_state->steps.front().job)->bytes !=
            expected_view_buffer))) {
    return 8;
  }
  if (backend == Backend::Vulkan) {
    const auto &binds =
        reduced_state->steps.front().job->workspace->arena->binds;
    const auto &input = binds.refs()[0u];
    const std::uint64_t input_end =
        input.offset_bytes + input.count * input.element_bytes;
    if (input.offset_bytes % view_alignment != 0u ||
        input.offset_bytes >= input_end ||
        input.bytes != expected_view_buffer) {
      return 8;
    }
    if (expected_bindings == 2u) {
      const auto &output = binds.refs()[1u];
      const std::uint64_t output_end =
          output.offset_bytes + output.count * output.element_bytes;
      if (binds.handles()[0u].get() != binds.handles()[1u].get() ||
          output.offset_bytes % view_alignment != 0u ||
          output.offset_bytes >= output_end ||
          !(input_end <= output.offset_bytes ||
            output_end <= input.offset_bytes) ||
          output.bytes != expected_view_buffer) {
        return 8;
      }
    }
  }
  const Status reduced_run = reduced->run();
  if (!reduced_run) {
    return 9;
  }
  std::array<std::int32_t, 3u> reduced_values{};
  const Stats first = reduced->stats();
  const std::uint64_t expected_reduce_roundtrip =
      backend == Backend::Vulkan ? 20u : 16u;
  const std::uint64_t expected_reduce_dispatches = backend == Backend::Cpu ? 1u
                                                   : backend == Backend::Metal
                                                       ? 2u
                                                       : 3u;
  if (!ReadExact(*reduced, *reduced_target, reduced_values) ||
      reduced_values != std::array<std::int32_t, 3u>{0, 16, 0} ||
      first.internal_roundtrip_bytes != expected_reduce_roundtrip ||
      first.dispatches != expected_reduce_dispatches ||
      (backend != Backend::Cpu && first.original_dispatches != 1u) ||
      (backend != Backend::Cpu &&
       first.final_dispatches != expected_reduce_dispatches)) {
    std::fprintf(
        stderr,
        "pipeline view reduce backend=%u roundtrip=%llu "
        "original=%llu dispatches=%llu final=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(first.internal_roundtrip_bytes),
        static_cast<unsigned long long>(first.original_dispatches),
        static_cast<unsigned long long>(first.dispatches),
        static_cast<unsigned long long>(first.final_dispatches));
    return 10;
  }
  std::array<MemoryEntry, 64u> before_entries{};
  const MemorySnapshot before = reduced->memory_snapshot(before_entries);
  if (backend == Backend::Cpu) {
    node_compute_allocation::Start();
  }
  const Status warm = reduced->run();
  if (backend == Backend::Cpu) {
    node_compute_allocation::Stop();
  }
  std::array<MemoryEntry, 64u> after_entries{};
  const MemorySnapshot after = reduced->memory_snapshot(after_entries);
  if (!warm ||
      (backend == Backend::Cpu && node_compute_allocation::Count() != 0u) ||
      before.truncated() || after.truncated() || before.total != after.total ||
      before.written != after.written ||
      !SameMemoryEntries(
          std::span<const MemoryEntry>{before_entries.data(), before.written},
          std::span<const MemoryEntry>{after_entries.data(), after.written}) ||
      !SameMemory(before.summary, after.summary)) {
    return 11;
  }
  auto shared_target = device.buffer<std::int32_t>(3u);
  if (!shared_target) {
    return 11;
  }
  auto shared_output = shared_target->view(1u, 1u, 2u);
  if (!shared_output) {
    return 11;
  }
  auto shared_builder =
      pipeline(device)
          .then(*reduce, read(*reduce_input), write(*reduce_output))
          .then(*reduce, read(*reduce_input), write(*shared_output));
  const auto shared_plan = shared_builder.plan();
  if (!shared_plan) {
    return 11;
  }
  const std::uint64_t expected_shared_buffer =
      backend == Backend::Cpu ? 2u * input_bytes + shared_plan->scratch_bytes
                              : reduced_plan->prepared_buffer_bytes;
  if (shared_plan->prepared_buffer_bytes != expected_shared_buffer ||
      shared_plan->prepared_host_bytes <= reduced_plan->prepared_host_bytes) {
    return 11;
  }
  auto shared_views = std::move(shared_builder).prepare();
  const std::shared_ptr<detail::PipelineState> shared_state =
      shared_views ? detail::PipelineStateAccess::state(*shared_views)
                   : std::shared_ptr<detail::PipelineState>{};
  if (shared_state == nullptr || shared_state->steps.size() != 2u ||
      shared_state->prepared_buffers.size() !=
          expected_owners + shared_plan->scratch_count ||
      (expected_owners != 0u &&
       (shared_state->steps[0u].job == nullptr ||
        shared_state->steps[1u].job == nullptr ||
        shared_state->steps[0u].job->workspace == nullptr ||
        shared_state->steps[1u].job->workspace == nullptr ||
        shared_state->steps[0u].job->workspace->arena == nullptr ||
        shared_state->steps[0u].job->workspace->arena !=
            shared_state->steps[1u].job->workspace->arena ||
        shared_state->steps[0u].job->workspace->arena->binds.size() !=
            expected_bindings + shared_plan->scratch_count))) {
    return 11;
  }
  if (backend == Backend::Cpu &&
      (shared_state->steps[0u].job == nullptr ||
       shared_state->steps[1u].job == nullptr ||
       CpuViewTransferCount(*shared_state->steps[0u].job) != 1u ||
       CpuViewTransferCount(*shared_state->steps[1u].job) != 1u ||
       FirstCpuViewBuffer(*shared_state->steps[0u].job) == nullptr ||
       FirstCpuViewBuffer(*shared_state->steps[0u].job) ==
           FirstCpuViewBuffer(*shared_state->steps[1u].job))) {
    return 11;
  }

  auto contiguous_target = device.buffer<std::int32_t>(3u);
  auto contiguous_input = source->view(2u, 4u);
  auto contiguous_output = contiguous_target->view(2u, 1u);
  if (!contiguous_target || !contiguous_input || !contiguous_output) {
    return 12;
  }
  auto contiguous =
      pipeline(device)
          .then(*reduce, read(*contiguous_input), write(*contiguous_output))
          .prepare();
  std::array<std::int32_t, 3u> contiguous_values{};
  const bool contiguous_dense =
      backend == Backend::Vulkan &&
      (2u * sizeof(std::int32_t)) % view_alignment != 0u;
  const std::uint64_t contiguous_dispatches = contiguous_dense ? 2u : 0u;
  const std::uint64_t contiguous_roundtrip =
      contiguous_dense ? 5u * sizeof(std::int32_t) : 0u;
  if (!contiguous || !contiguous->run() ||
      !ReadExact(*contiguous, *contiguous_target, contiguous_values) ||
      contiguous_values != std::array<std::int32_t, 3u>{0, 0, 14} ||
      contiguous->stats().internal_roundtrip_bytes != contiguous_roundtrip ||
      (backend != Backend::Cpu &&
       (contiguous->stats().dispatches !=
            contiguous->stats().final_dispatches ||
        contiguous->stats().final_dispatches !=
            contiguous->stats().original_dispatches + contiguous_dispatches))) {
    return 13;
  }

  return 0;
}

} // namespace rund_node_test_pipeline::view
