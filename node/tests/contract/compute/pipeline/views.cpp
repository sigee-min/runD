#include "local.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/math.hpp>

#include "../allocation.hpp"
#include "src/compute/pipeline/plan/arena.hpp"
#include "src/compute/pipeline/state.hpp"

#include <cstdio>
#include <limits>

namespace rund_node_test_pipeline {
namespace {

[[nodiscard]] std::size_t
CpuViewTransferCount(const rund::compute::detail::JobState &job) noexcept {
  return job.cpu_view_inputs.size() + job.cpu_view_outputs.size();
}

[[nodiscard]] const rund::compute::detail::BufferState *
FirstCpuViewBuffer(const rund::compute::detail::JobState &job) noexcept {
  if (!job.cpu_view_inputs.empty()) {
    const auto &transfer = job.cpu_view_inputs.front();
    return transfer.binding < job.inputs.size()
               ? job.inputs[transfer.binding].get()
               : nullptr;
  }
  if (!job.cpu_view_outputs.empty()) {
    const auto &transfer = job.cpu_view_outputs.front();
    return transfer.binding < job.outputs.size()
               ? job.outputs[transfer.binding].get()
               : nullptr;
  }
  return nullptr;
}

template <class T>
[[nodiscard]] bool CheckStridedReset(rund::compute::Device &device,
                                     const Backend backend) {
  using namespace rund::compute;
  auto scatter =
      on(device)
          .map<T>("pipeline-view-reset", 1u, [](auto value) { return value; })
          .scatter(1u, {.count = 2u})
          .compile();
  constexpr std::array<T, 4u> history{101u, 202u, 303u, 404u};
  constexpr std::array<T, 1u> first_value{7u};
  constexpr std::array<std::uint32_t, 1u> first_index{0u};
  constexpr std::array<T, 1u> second_value{9u};
  constexpr std::array<std::uint32_t, 1u> second_index{1u};
  auto target = device.upload<T>(history);
  auto first_values = device.upload<T>(first_value);
  auto first_indices = device.upload<std::uint32_t>(first_index);
  auto second_values = device.upload<T>(second_value);
  auto second_indices = device.upload<std::uint32_t>(second_index);
  if (!scatter || !target || !first_values || !first_indices ||
      !second_values || !second_indices ||
      scatter->graph().memory.reset_bytes != 2u * sizeof(T) ||
      scatter->graph().memory.reset_count != 1u) {
    return false;
  }
  auto view = target->view(0u, 2u, 2u);
  if (!view) {
    return false;
  }
  auto first =
      pipeline(device)
          .then(*scatter, read(*first_values, *first_indices), write(*view))
          .prepare();
  std::array<T, 4u> observed{};
  if (!first || !first->run() || !ReadExact(*first, *target, observed) ||
      observed != std::array<T, 4u>{7u, 202u, 0u, 404u}) {
    return false;
  }
  auto second =
      pipeline(device)
          .then(*scatter, read(*second_values, *second_indices), write(*view))
          .prepare();
  if (!second || !second->run() || !ReadExact(*second, *target, observed) ||
      observed != std::array<T, 4u>{0u, 202u, 9u, 404u} ||
      second->stats().reset_bytes != 2u * sizeof(T) ||
      second->stats().reset_commands != 1u) {
    std::fprintf(
        stderr,
        "pipeline reset backend=%u width=%zu bytes=%llu commands=%llu\n",
        static_cast<unsigned>(backend), sizeof(T),
        static_cast<unsigned long long>(second ? second->stats().reset_bytes
                                               : 0u),
        static_cast<unsigned long long>(second ? second->stats().reset_commands
                                               : 0u));
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckScatterOffset(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 5u> values{99, 5, 7, 11, 13};
  constexpr std::array<std::uint32_t, 5u> indices{99u, 0u, 1u, 0u, 1u};
  constexpr std::array<std::uint32_t, 3u> counts{99u, 4u, 99u};
  auto value_buffer = Upload(device, values);
  auto index_buffer = Upload(device, indices);
  auto count_buffer = Upload(device, counts);
  auto output_buffer = device.buffer<std::int32_t>(4u);
  auto exact = on(device)
                   .input<std::int32_t>(4u)
                   .zip_input<std::uint32_t>(4u)
                   .branch([](auto input, auto targets) {
                     return input.scatter_reduce(targets, 2u, Reduce::Sum);
                   })
                   .compile();
  auto bounded = on(device)
                     .input<Bounded<std::int32_t>>(4u)
                     .branch([](auto input) {
                       auto targets = input.indices().map(
                           "pipeline-scatter-target", [](auto ordinal) {
                             return ordinal & std::uint32_t{1};
                           });
                       return input.scatter_reduce(targets, 2u, Reduce::Sum);
                     })
                     .compile();
  if (!value_buffer || !index_buffer || !count_buffer || !output_buffer ||
      !exact || !bounded) {
    return false;
  }
  auto value_view = value_buffer->view(1u, 4u);
  auto index_view = index_buffer->view(1u, 4u);
  auto count_view = count_buffer->view(1u, 1u);
  auto output_view = output_buffer->view(1u, 2u);
  if (!value_view || !index_view || !count_view || !output_view) {
    return false;
  }
  auto exact_run =
      pipeline(device)
          .then(*exact, read(*value_view, *index_view), write(*output_view))
          .prepare();
  std::array<std::int32_t, 4u> observed{};
  if (!exact_run || !exact_run->run() ||
      !ReadExact(*exact_run, *output_buffer, observed) ||
      observed != std::array<std::int32_t, 4u>{0, 16, 20, 0}) {
    return false;
  }
  auto bounded_run =
      pipeline(device)
          .then(*bounded, read(*value_view, *count_view), write(*output_view))
          .prepare();
  return bounded_run && bounded_run->run() &&
         ReadExact(*bounded_run, *output_buffer, observed) &&
         observed == std::array<std::int32_t, 4u>{0, 16, 20, 0};
}

[[nodiscard]] bool CheckNestedCpuViewPhase(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> initial_values{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 8u> backing_values{99u, 1u, 99u, 2u,
                                                         99u, 3u, 99u, 4u};
  constexpr std::array<std::uint32_t, 1u> count_value{2u};
  auto seed = on(device)
                  .input<std::uint32_t>(4u)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto external, auto total, auto ordinal) {
                    (void)total;
                    auto sum = external.reduce(Reduce::Sum);
                    return sum.combine(
                        "pipeline-view-nested-seed", ordinal.scalar(),
                        [](auto value, auto outer) { return value + outer; });
                  })
                  .compile();
  auto fold =
      on(device)
          .input<std::uint32_t>(4u)
          .zip_input<std::uint32_t>(1u)
          .branch([](auto outer, auto tile) {
            return outer.combine(
                "pipeline-view-nested-fold", tile.scalar(),
                [](auto value, auto increment) { return value + increment; });
          })
          .compile();
  auto initial = device.upload<std::uint32_t>(initial_values);
  auto backing = device.upload<std::uint32_t>(backing_values);
  auto count = device.upload<std::uint32_t>(count_value);
  auto target = device.buffer<std::uint32_t>(4u);
  if (!seed || !fold || !initial || !backing || !count || !target) {
    return false;
  }
  auto external = backing->view(1u, 4u, 2u);
  if (!external) {
    return false;
  }
  // Fold consumes the sealed dense recurrent banks, so the reachable CPU View
  // transfer is the Seed-only external input. This proves record_cpu_view uses
  // the common route projector while retaining the compact Seed outer ordinal.
  const auto body = tile_repeat<0u>(*seed, *fold);
  auto builder = pipeline(device);
  builder.windows<2u, 1u>(body, window(*count), read(*initial, *external),
                          write_final(*target));
  const auto plan = builder.plan();
  if (!plan || plan->view_nested_phase != PipelineNestedPhase::Seed ||
      plan->view_outer_window != 0u ||
      plan->view_inner_iteration != std::numeric_limits<std::size_t>::max() ||
      plan->view_count != 4u ||
      plan->view_stride_bytes != 2u * sizeof(std::uint32_t) ||
      plan->view_span_bytes != 7u * sizeof(std::uint32_t)) {
    std::fprintf(
        stderr,
        "nested CPU view plan status=%u reason=%u phase=%u step=%llu "
        "iteration=%llu coordinates=%llu/%llu count=%llu stride=%llu "
        "span=%llu\n",
        static_cast<unsigned>(plan.ok()), static_cast<unsigned>(plan.reason()),
        static_cast<unsigned>(plan ? plan->view_nested_phase
                                   : PipelineNestedPhase::None),
        static_cast<unsigned long long>(plan ? plan->view_step : 0u),
        static_cast<unsigned long long>(plan ? plan->view_iteration : 0u),
        static_cast<unsigned long long>(plan ? plan->view_outer_window : 0u),
        static_cast<unsigned long long>(plan ? plan->view_inner_iteration : 0u),
        static_cast<unsigned long long>(plan ? plan->view_count : 0u),
        static_cast<unsigned long long>(plan ? plan->view_stride_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->view_span_bytes : 0u));
    return false;
  }
  auto prepared = std::move(builder).prepare();
  std::array<std::uint32_t, 4u> observed{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const bool read = prepared && ReadExact(*prepared, *target, observed);
  if (!prepared || !ran || !read ||
      observed != std::array<std::uint32_t, 4u>{22u, 23u, 24u, 25u}) {
    std::fprintf(stderr,
                 "nested CPU view run prepared=%u/%u run=%u/%u read=%u "
                 "values=%u,%u,%u,%u\n",
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()),
                 static_cast<unsigned>(ran.ok()),
                 static_cast<unsigned>(ran.reason()),
                 static_cast<unsigned>(read), observed[0u], observed[1u],
                 observed[2u], observed[3u]);
    return false;
  }
  return true;
}

} // namespace

[[nodiscard]] int CheckViews(rund::compute::Device &device,
                             const Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu && !CheckNestedCpuViewPhase(device)) {
    return 9;
  }
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto source = Upload(device, source_values);
  auto target = device.buffer<std::int32_t>(source_values.size());
  auto program = on(device)
                     .map<std::int32_t>("pipeline-strided-view", 3u,
                                        [](auto value) { return value * 10; })
                     .compile();
  if (!source || !target || !program) {
    return 1;
  }
  auto input_view = source->view(1u, 3u, 2u);
  auto output_view = target->view(0u, 3u, 2u);
  if (!input_view || !output_view || input_view->offset() != 1u ||
      input_view->size() != 3u || input_view->stride() != 2u ||
      input_view->span_bytes() != 5u * sizeof(std::int32_t)) {
    return 2;
  }
  auto prepared = pipeline(device)
                      .then(*program, read(*input_view), write(*output_view))
                      .prepare();
  if (!prepared) {
    return 3;
  }
  const Status view_run = prepared->run();
  if (!view_run || prepared->stats().pipeline.barrier_count != 0u) {
    return 3;
  }
  std::array<std::int32_t, source_values.size()> observed{};
  if (!ReadExact(*prepared, *target, observed) ||
      observed != std::array<std::int32_t, 8u>{10, 0, 30, 0, 50, 0, 0, 0}) {
    return 4;
  }

  auto even = on(device)
                  .map<std::int32_t>("pipeline-even-view", 4u,
                                     [](auto value) { return value + 1; })
                  .compile();
  auto odd = on(device)
                 .map<std::int32_t>("pipeline-odd-view", 4u,
                                    [](auto value) { return value + 2; })
                 .compile();
  auto disjoint_target = device.buffer<std::int32_t>(source_values.size());
  auto even_input = source->view(0u, 4u, 2u);
  auto odd_input = source->view(1u, 4u, 2u);
  auto even_output = disjoint_target->view(0u, 4u, 2u);
  auto odd_output = disjoint_target->view(1u, 4u, 2u);
  if (!even || !odd || !disjoint_target || !even_input || !odd_input ||
      !even_output || !odd_output) {
    return 5;
  }
  auto disjoint = pipeline(device)
                      .then(*even, read(*even_input), write(*even_output))
                      .then(*odd, read(*odd_input), write(*odd_output))
                      .prepare();
  if (!disjoint || !disjoint->run() ||
      disjoint->stats().pipeline.barrier_count != 0u ||
      !ReadExact(*disjoint, *disjoint_target, observed) ||
      observed != std::array<std::int32_t, 8u>{1, 3, 3, 5, 5, 7, 7, 9}) {
    return 6;
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

  constexpr std::array<std::int32_t, 8u> unsorted_values{9, 4, 7, 2,
                                                         5, 8, 1, 6};
  auto unsorted = Upload(device, unsorted_values);
  auto sort_target = device.buffer<std::int32_t>(unsorted_values.size());
  auto sort = on(device)
                  .input<std::int32_t>(4u)
                  .branch([](auto values) { return values.sort(); })
                  .compile();
  if (!unsorted || !sort_target || !sort) {
    return 14;
  }
  auto sort_input = unsorted->view(1u, 4u, 2u);
  auto sort_output = sort_target->view(0u, 4u, 2u);
  if (!sort_input || !sort_output) {
    return 15;
  }
  auto sorted = pipeline(device)
                    .then(*sort, read(*sort_input), write(*sort_output))
                    .prepare();
  std::array<std::int32_t, 8u> sorted_values{};
  if (!sorted || !sorted->run() ||
      !ReadExact(*sorted, *sort_target, sorted_values) ||
      sorted_values != std::array<std::int32_t, 8u>{2, 0, 4, 0, 6, 0, 8, 0} ||
      sorted->stats().internal_roundtrip_bytes != 32u ||
      (backend != Backend::Cpu &&
       (sorted->stats().dispatches != sorted->stats().final_dispatches ||
        sorted->stats().final_dispatches !=
            sorted->stats().original_dispatches + 2u))) {
    if (sorted) {
      std::fprintf(
          stderr,
          "pipeline view sort backend=%u roundtrip=%llu original=%llu "
          "dispatches=%llu final=%llu first=%d second=%d\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(
              sorted->stats().internal_roundtrip_bytes),
          static_cast<unsigned long long>(sorted->stats().original_dispatches),
          static_cast<unsigned long long>(sorted->stats().dispatches),
          static_cast<unsigned long long>(sorted->stats().final_dispatches),
          sorted_values[0u], sorted_values[2u]);
    } else {
      std::fprintf(stderr, "pipeline view sort backend=%u prepare=%u\n",
                   static_cast<unsigned>(backend),
                   static_cast<unsigned>(sorted.reason()));
    }
    return 16;
  }

  // Return an exact-size nonzero allocation at a logical ownership boundary.
  // Pooling backends reacquire that storage; every backend must still publish
  // a zeroed Buffer before the partial strided write touches only even lanes.
  auto poison =
      on(device)
          .map<std::int32_t>("pipeline-view-pool-poison", source_values.size(),
                             [](auto value) { return value + 101; })
          .compile();
  if (!poison) {
    return 17;
  }
  {
    auto poison_target = device.buffer<std::int32_t>(source_values.size());
    if (!poison_target) {
      return 18;
    }
    auto poisoned = pipeline(device)
                        .then(*poison, read(*source), write(*poison_target))
                        .prepare();
    if (!poisoned || !poisoned->run()) {
      return 19;
    }
  }
  auto reused_target = device.buffer<std::int32_t>(source_values.size());
  if (!reused_target) {
    return 20;
  }
  auto reused_output = reused_target->view(0u, 4u, 2u);
  if (!reused_output) {
    return 21;
  }
  auto reused = pipeline(device)
                    .then(*even, read(*even_input), write(*reused_output))
                    .prepare();
  if (!reused || !reused->run() ||
      !ReadExact(*reused, *reused_target, observed) ||
      observed != std::array<std::int32_t, 8u>{1, 0, 3, 0, 5, 0, 7, 0}) {
    return 22;
  }

  auto scan =
      on(device)
          .input<std::int32_t>(4u)
          .branch([](auto values) { return values.scan(Scan::InclusiveSum); })
          .compile();
  auto scan_target = device.buffer<std::int32_t>(source_values.size());
  auto scan_input = source->view(1u, 4u, 2u);
  if (!scan || !scan_target || !scan_input) {
    return 23;
  }
  auto scan_output = scan_target->view(1u, 4u, 2u);
  if (!scan_output) {
    return 23;
  }
  auto scanned = pipeline(device)
                     .then(*scan, read(*scan_input), write(*scan_output))
                     .prepare();
  std::array<std::int32_t, 8u> scanned_values{};
  if (!scanned || !scanned->run() ||
      !ReadExact(*scanned, *scan_target, scanned_values) ||
      scanned_values != std::array<std::int32_t, 8u>{0, 1, 0, 4, 0, 9, 0, 16} ||
      scanned->stats().internal_roundtrip_bytes != 32u ||
      (backend != Backend::Cpu &&
       (scanned->stats().dispatches != scanned->stats().final_dispatches ||
        scanned->stats().final_dispatches !=
            scanned->stats().original_dispatches + 2u))) {
    return 24;
  }

  constexpr std::array<std::int32_t, 8u> left_values{1, 0, 2, 0, 3, 0, 4, 0};
  constexpr std::array<std::int32_t, 8u> right_values{5, 0, 6, 0, 7, 0, 8, 0};
  auto left = Upload(device, left_values);
  auto right = Upload(device, right_values);
  auto matrix_target = device.buffer<std::int32_t>(left_values.size());
  auto matrix = on(device)
                    .map<std::int32_t>("pipeline-view-matrix", 4u,
                                       [](auto value) { return value; })
                    .matrix<2u, 2u>()
                    .matmul<2u, 2u>()
                    .compile();
  if (!left || !right || !matrix_target || !matrix) {
    return 25;
  }
  auto left_view = left->view(0u, 4u, 2u);
  auto right_view = right->view(0u, 4u, 2u);
  auto matrix_output = matrix_target->view(0u, 4u, 2u);
  if (!left_view || !right_view || !matrix_output) {
    return 26;
  }
  auto multiplied =
      pipeline(device)
          .then(*matrix, read(*left_view, *right_view), write(*matrix_output))
          .prepare();
  std::array<std::int32_t, 8u> matrix_values{};
  if (!multiplied || !multiplied->run() ||
      !ReadExact(*multiplied, *matrix_target, matrix_values) ||
      matrix_values !=
          std::array<std::int32_t, 8u>{19, 0, 22, 0, 43, 0, 50, 0} ||
      multiplied->stats().internal_roundtrip_bytes != 48u ||
      (backend != Backend::Cpu &&
       (multiplied->stats().dispatches !=
            multiplied->stats().final_dispatches ||
        multiplied->stats().final_dispatches !=
            multiplied->stats().original_dispatches + 3u))) {
    return 27;
  }

  if (const int arena = CheckViewArena(device, backend); arena != 0) {
    return 30 + arena;
  }

  // A reset route bound to a strided external View resets the authored View,
  // not its complete Buffer owner. The second Pipeline starts from history
  // written by the first and must clear both View lanes before scattering.
  if (!CheckStridedReset<std::uint32_t>(device, backend) ||
      !CheckStridedReset<std::uint64_t>(device, backend) ||
      !CheckScatterOffset(device)) {
    return 28;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
