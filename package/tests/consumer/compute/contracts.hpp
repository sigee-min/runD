#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <memory_resource>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

template <class T, std::size_t N> struct SdkContiguousInput final {
  T values[N]{};

  [[nodiscard]] constexpr T *begin() noexcept { return values; }
  [[nodiscard]] constexpr const T *begin() const noexcept { return values; }
  [[nodiscard]] constexpr T *end() noexcept { return values + N; }
  [[nodiscard]] constexpr const T *end() const noexcept { return values + N; }
  [[nodiscard]] constexpr T *data() noexcept { return values; }
  [[nodiscard]] constexpr const T *data() const noexcept { return values; }
  [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
};

template <class T>
concept HasRemainder = requires(T value) { value % value; };
template <class T>
concept HasWorkers = requires(T value) {
  { value.workers() } -> std::same_as<std::uint32_t>;
};
template <class T>
concept AcceptsRvalueFlowInput = requires(T value) {
  rund::compute::on(rund::compute::Target::cpu(), std::move(value));
};
template <class Range>
concept AcceptsFlowRange = requires(Range &value) {
  rund::compute::on(rund::compute::Target::cpu(), value);
};
template <class... Args>
concept MakesComputeFailure = requires(Args &&...args) {
  rund::compute::Status::fail(std::forward<Args>(args)...);
};
template <class Range>
concept AcceptsConstFlowRange = requires(const Range &value) {
  rund::compute::on(rund::compute::Target::cpu(), value);
};
template <class T>
concept AcceptsBufferValue = requires(const rund::compute::Device &device) {
  device.template buffer<T>(1u);
};
template <class T>
concept CastsToFloat = requires(T value) { static_cast<float>(value); };
template <class T>
concept CastsToDouble = requires(T value) { static_cast<double>(value); };
template <unsigned IntegerBits, unsigned FractionBits>
concept DefinesFixed =
    requires { typename rund::compute::Fixed<IntegerBits, FractionBits>; };
template <class T>
concept AddsFloatExpression =
    requires(rund::compute::Expr<T> value) { value + 1.0f; };
template <class Left, class Right>
concept AddsExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left + right;
    };
template <class Left, class Right>
concept SubtractsExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left - right;
    };
template <class Left, class Right>
concept MultipliesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left * right;
    };
template <class Left, class Right>
concept DividesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left / right;
    };
template <class Left, class Right>
concept MinimizesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      rund::compute::min(left, right);
    };
template <class Left, class Right>
concept MaximizesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      rund::compute::max(left, right);
    };
template <class T>
concept HasFixedDivide =
    requires(const T &value) { rund::compute::div_fixed(value, value); };
template <class T>
concept HasUnsignedStorageSaturate =
    requires(const T &value) { rund::compute::add_sat_unsigned(value, value); };
template <class T>
concept HasArithmeticShift =
    requires(const T &value) { rund::compute::shr_arithmetic<1u>(value); };
template <class T>
concept HasProgramObservers =
    requires(const T &program, std::span<rund::compute::MemoryEntry> entries) {
      { program.valid() } -> std::same_as<bool>;
      {
        program.backend()
      } -> std::same_as<rund::compute::Result<rund::compute::Backend>>;
      { program.graph() } -> std::same_as<const rund::compute::graph::Info &>;
      {
        program.fingerprint()
      } -> std::same_as<rund::compute::graph::Fingerprint>;
      { program.memory() } -> std::same_as<rund::compute::MemoryStats>;
      {
        program.memory_snapshot(entries)
      } -> std::same_as<rund::compute::MemorySnapshot>;
    };
template <class T>
concept HasComputeTelemetry = requires(T stats) {
  stats.pipeline_compiles;
  stats.buffer_allocations;
  stats.download_events;
  stats.dispatches;
  stats.command_submits;
  stats.uploaded_bytes;
  stats.downloaded_bytes;
  stats.pipeline_cache_hits;
  stats.pipeline_cache_evictions;
  stats.buffer_reuses;
  stats.descriptor_pool_creations;
  stats.descriptor_set_allocations;
  stats.descriptor_reuses;
  stats.original_dispatches;
  stats.final_dispatches;
  stats.fusions;
  stats.fusion_rejections;
  stats.internal_roundtrip_bytes;
  stats.external_roundtrip_bytes;
  stats.reset_bytes;
  stats.reset_commands;
  stats.kernel_ns;
  stats.kernel_samples;
  stats.shader_compile_ns;
  stats.spirv_compile_ns;
  stats.pipeline_create_ns;
  stats.descriptor_setup_ns;
  stats.submit_wait_ns;
  stats.readback_ns;
  stats.pipeline.step_count;
  stats.pipeline.resource_count;
  stats.pipeline.barrier_count;
  stats.pipeline.claim_conflict_count;
  stats.pipeline.verified_step_count;
  stats.pipeline.failed_step_index;
  stats.pipeline.status_entry_count;
  stats.pipeline.control_byte_count;
  stats.pipeline.control_command_count;
  stats.pipeline.claim_ns;
  stats.pipeline.control_ns;
  { stats.available() } -> std::same_as<bool>;
  { stats.kernel_timing_available() } -> std::same_as<bool>;
};
template <class T>
concept ConfiguresPipelineProfile = requires(T &builder) {
  {
    builder.profile(rund::compute::PipelineProfile::Steps)
  } -> std::same_as<T &>;
  {
    std::move(builder).profile(rund::compute::PipelineProfile::Steps)
  } -> std::same_as<T &&>;
};
template <class T>
concept PlansPipeline = requires(T &builder) {
  {
    builder.plan()
  } -> std::same_as<rund::compute::Result<rund::compute::PipelinePlan>>;
  {
    builder.budget(rund::compute::MemoryBudget{.bytes = 1u})
  } -> std::same_as<T &>;
};
template <class T>
concept HasPipelinePlan = requires(const T &plan) {
  plan.persistent_bytes;
  plan.state_bytes;
  plan.transient_bytes;
  plan.prepared_bytes;
  plan.publish_bytes;
  plan.peak_bytes;
  plan.total_bytes;
  plan.allocation_count;
  plan.reuse_count;
  plan.publish_count;
  plan.largest_bytes;
  plan.largest_step;
  plan.largest_iteration;
  plan.largest_chunk;
  plan.view_bytes;
  plan.view_span_bytes;
  plan.view_backing_bytes;
  plan.view_offset_bytes;
  plan.view_stride_bytes;
  plan.view_element_bytes;
  plan.view_count;
  plan.view_alignment;
  plan.view_step;
  plan.view_iteration;
  plan.view_binding;
  plan.peak_step;
  plan.peak_iteration;
};
template <class T>
concept HasPipelineProfile = requires(
    const T &pipeline, std::span<rund::compute::PipelineStepProfile> steps) {
  {
    pipeline.profile(steps)
  } -> std::same_as<
      rund::compute::Result<rund::compute::PipelineProfileSnapshot>>;
};
template <class T>
concept HasPipelineStepStats = requires(const T &stats) {
  stats.sample_count;
  stats.original_dispatches;
  stats.final_dispatches;
  stats.barrier_count;
  stats.tile_count;
  stats.tile_size;
  stats.vector_chunks;
  stats.tail_chunks;
  stats.workgroup_count;
  stats.work_item_count;
  stats.control;
  stats.worker_count;
  stats.participating_workers;
  { stats.available() } -> std::same_as<bool>;
};
template <class T>
concept HasStepTiming = requires(const T &timing) {
  timing.duration_ns;
  timing.sample_count;
  timing.clock;
  timing.relation;
  { timing.available() } -> std::same_as<bool>;
  { timing.saturated() } -> std::same_as<bool>;
};
template <class T>
concept HasPipelineStepProfile = requires(const T &step) {
  step.index;
  step.iteration;
  step.iteration_bound;
  step.program;
  step.timing;
  step.execution;
  step.memory;
};
template <class T>
concept HasPipelineProfileSnapshot = requires(const T &snapshot) {
  snapshot.execution;
  snapshot.memory;
  snapshot.shared_memory;
  snapshot.observation;
  snapshot.referenced_resource_bytes;
  snapshot.instrumentation_command_count;
  snapshot.instrumentation_byte_count;
  snapshot.written;
  snapshot.total;
  { snapshot.truncated() } -> std::same_as<bool>;
};
static_assert(HasComputeTelemetry<rund::compute::Stats>);
static_assert(ConfiguresPipelineProfile<rund::compute::PipelineBuilder>);
static_assert(PlansPipeline<rund::compute::PipelineBuilder>);
static_assert(HasPipelinePlan<rund::compute::PipelinePlan>);
static_assert(HasPipelineProfile<rund::compute::Pipeline>);
static_assert(HasPipelineStepStats<rund::compute::PipelineStepStats>);
static_assert(HasStepTiming<rund::compute::StepTiming>);
static_assert(HasPipelineStepProfile<rund::compute::PipelineStepProfile>);
static_assert(
    HasPipelineProfileSnapshot<rund::compute::PipelineProfileSnapshot>);
static_assert(rund::compute::PipelineStats{}.failed_step_index ==
              rund::compute::PipelineStats::no_failed_step);
static_assert(!rund::compute::Stats{}.available());
static_assert(rund::compute::Stats{}.backend ==
              rund::compute::Backend::Unavailable);
static_assert(rund::compute::PipelineProfile{} ==
              rund::compute::PipelineProfile::None);
static_assert(!rund::compute::StepTiming{}.available());
static_assert(!rund::compute::PipelineStepStats{}.available());
constexpr rund::compute::StepTiming ComputeMeasuredZeroStep{
    .duration_ns = 0u,
    .sample_count = 1u,
    .clock = rund::compute::StepClock::HostSteady,
    .relation = rund::compute::StepTimingRelation::Exclusive,
};
constexpr rund::compute::PipelineStepStats ComputeMeasuredZeroWork{
    .sample_count = 1u,
};
constexpr rund::compute::PipelineProfileSnapshot
    ComputeTruncatedPipelineProfile{.written = 1u, .total = 2u};
static_assert(ComputeMeasuredZeroStep.available());
static_assert(!ComputeMeasuredZeroStep.saturated());
static_assert(ComputeMeasuredZeroWork.available());
static_assert(ComputeTruncatedPipelineProfile.truncated());

constexpr rund::compute::Status ComputeStatusSuccess =
    rund::compute::Status::success();
constexpr rund::compute::Status ComputeStatusBindingFailure =
    rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
constexpr rund::compute::Status ComputeStatusDeviceBusy =
    rund::compute::Status::fail(rund::compute::Reason::DeviceBusy);
constexpr rund::compute::Status ComputeProfileUnavailable =
    rund::compute::Status::fail(rund::compute::Reason::ProfileUnavailable);
static_assert(ComputeStatusSuccess.ok());
static_assert(ComputeStatusSuccess.code() == rund::compute::Code::Ok);
static_assert(ComputeStatusSuccess.exit_code() == 0);
static_assert(!ComputeStatusBindingFailure.ok());
static_assert(ComputeStatusBindingFailure.code() ==
              rund::compute::Code::Binding);
static_assert(ComputeStatusBindingFailure.reason() ==
              rund::compute::Reason::ShapeMismatch);
static_assert(ComputeStatusBindingFailure.exit_code() == 1);
static_assert(!ComputeStatusDeviceBusy.ok());
static_assert(ComputeStatusDeviceBusy.code() == rund::compute::Code::Execution);
static_assert(ComputeStatusDeviceBusy.reason() ==
              rund::compute::Reason::DeviceBusy);
static_assert(!ComputeProfileUnavailable.ok());
static_assert(ComputeProfileUnavailable.reason() ==
              rund::compute::Reason::ProfileUnavailable);
static_assert(ComputeProfileUnavailable.code() ==
              rund::compute::Code::Unavailable);
constexpr rund::compute::Status ComputeStatusInvalidReason =
    rund::compute::Status::fail(static_cast<rund::compute::Reason>(0xffffu));
static_assert(!ComputeStatusInvalidReason.ok());
static_assert(ComputeStatusInvalidReason.reason() ==
              rund::compute::Reason::ReasonInvalid);
static_assert(ComputeStatusInvalidReason.code() ==
              rund::compute::Code::Invalid);
static_assert(!MakesComputeFailure<rund::compute::Code, std::string_view>);
static_assert(MakesComputeFailure<rund::compute::Reason>);
static_assert(sizeof(rund::compute::Status) == 2u);
constexpr rund::compute::Compile ComputeCompile{};
static_assert(ComputeCompile.workers == 2u);
static_assert(ComputeCompile.capacity == 64u);

static_assert(sizeof(rund::compute::Fixed<16, 16>) == sizeof(std::int32_t));
static_assert(sizeof(rund::compute::Fixed<20, 44>) == sizeof(std::int64_t));
static_assert(
    std::is_same_v<typename rund::compute::Fixed<16, 16>::Raw, std::int32_t>);
static_assert(
    std::is_same_v<typename rund::compute::Fixed<20, 44>::Raw, std::int64_t>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<16, 16>, float>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<16, 16>, double>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<20, 44>, float>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<20, 44>, double>);
static_assert(!CastsToFloat<rund::compute::Fixed<16, 16>>);
static_assert(!CastsToDouble<rund::compute::Fixed<16, 16>>);
static_assert(!CastsToFloat<rund::compute::Fixed<20, 44>>);
static_assert(!CastsToDouble<rund::compute::Fixed<20, 44>>);
static_assert(!std::constructible_from<rund::compute::Fixed<16, 16>, float>);
static_assert(!std::constructible_from<rund::compute::Fixed<32, 32>, double>);
static_assert(!DefinesFixed<std::numeric_limits<unsigned>::max(), 33u>);
static_assert(!AddsFloatExpression<rund::compute::Fixed<16, 16>>);
using Fixed16x16 = rund::compute::Fixed<16, 16>;
using Fixed1x31 = rund::compute::Fixed<1, 31>;
using Fixed8x24 = rund::compute::Fixed<8, 24>;
static_assert(!AddsExpressions<Fixed16x16, Fixed1x31>);
static_assert(!SubtractsExpressions<Fixed16x16, Fixed8x24>);
static_assert(!MultipliesExpressions<Fixed16x16, Fixed8x24>);
static_assert(!DividesExpressions<Fixed16x16, Fixed8x24>);
static_assert(!MinimizesExpressions<Fixed16x16, Fixed8x24>);
static_assert(!MaximizesExpressions<Fixed16x16, Fixed8x24>);

static_assert(!std::is_default_constructible_v<rund::compute::Device>);
static_assert(!std::is_copy_constructible_v<rund::compute::Device>);
static_assert(std::is_move_constructible_v<rund::compute::Device>);
static_assert(HasWorkers<rund::compute::Target>);
static_assert(!HasWorkers<rund::compute::Backend>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Resource{}.rounding),
                   rund::compute::Rounding>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Resource{}.overflow),
                   rund::compute::Overflow>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Resource{}.approximation),
                   rund::compute::Approximation>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Resource{}.active),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Resource{}.parent),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Resource{}.source),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Info{}.memory),
                             rund::compute::graph::MemoryPlan>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Info{}.authored_nodes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Info{}.lowered_nodes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.logical_bytes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.live_bytes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.physical_bytes),
                   std::uint64_t>);
static_assert(std::is_same_v<
              decltype(rund::compute::graph::MemoryPlan{}.allocation_count),
              std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.reset_bytes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.reset_count),
                   std::uint64_t>);
static_assert(std::is_same_v<decltype(rund::compute::Stats{}.reset_bytes),
                             std::uint64_t>);
static_assert(std::is_same_v<decltype(rund::compute::Stats{}.reset_commands),
                             std::uint64_t>);
static_assert(
    !std::is_default_constructible_v<rund::compute::Buffer<std::int32_t>>);
using ComputeDeviceResult = rund::compute::Result<rund::compute::Device>;
static_assert(
    std::is_same_v<decltype(rund::compute::open(rund::compute::Target::cpu())),
                   ComputeDeviceResult>);
static_assert(
    std::is_same_v<decltype(std::declval<ComputeDeviceResult &>().operator->()),
                   rund::compute::Device *>);
static_assert(std::is_same_v<decltype(*std::declval<ComputeDeviceResult &>()),
                             rund::compute::Device &>);
static_assert(std::same_as<
              decltype(std::declval<const ComputeDeviceResult &>().exit_code()),
              int>);
using IntJob = rund::compute::Job<std::int32_t(std::int32_t)>;
using IntProgram = rund::compute::Program<std::int32_t(std::int32_t)>;
using MultiProgram =
    rund::compute::Program<std::int32_t(std::int32_t, std::uint32_t)>;
using BoundedProgram =
    rund::compute::Program<rund::compute::Bounded<std::int32_t>(std::int32_t)>;
using OutputProgram =
    rund::compute::Program<rund::compute::Outputs<std::int32_t, std::uint32_t>(
        std::int32_t)>;
using IntFlow = rund::compute::Flow<std::int32_t(std::int32_t)>;
using IntStage =
    rund::compute::StageRef<std::int32_t, rund::compute::stage::Exact>;
using UintStage =
    rund::compute::StageRef<std::uint32_t, rund::compute::stage::Exact>;
using HeterogeneousZip = decltype(rund::compute::zip(
    std::declval<const IntStage &>(), std::declval<const UintStage &>()));
using HeterogeneousRecord = decltype(rund::compute::record(
    std::declval<const IntStage &>(), std::declval<const UintStage &>()));
template <class Range>
concept AcceptsGatherRange =
    requires(IntFlow flow, Range &value) { std::move(flow).gather(value); };
static_assert(
    std::same_as<decltype(rund::compute::on(rund::compute::Target::cpu())),
                 rund::compute::FlowBuilder>);
static_assert(std::same_as<
              decltype(rund::compute::on(rund::compute::Target::cpu(),
                                         std::declval<std::int32_t (&)[4]>())),
              IntFlow>);
static_assert(std::same_as<
              decltype(rund::compute::on(
                  rund::compute::Target::cpu(),
                  std::declval<const SdkContiguousInput<std::int32_t, 4> &>())),
              IntFlow>);
static_assert(
    std::same_as<decltype(rund::compute::on(
                     rund::compute::Target::cpu(),
                     std::declval<std::pmr::vector<std::int32_t> &>())),
                 IntFlow>);
static_assert(
    std::same_as<decltype(rund::compute::on(
                     rund::compute::Target::cpu(),
                     std::declval<const std::pmr::vector<std::int32_t> &>())),
                 IntFlow>);
static_assert(!std::is_copy_constructible_v<IntJob>);
static_assert(std::is_move_constructible_v<IntJob>);
static_assert(HasProgramObservers<IntProgram>);
static_assert(HasProgramObservers<MultiProgram>);
static_assert(HasProgramObservers<BoundedProgram>);
static_assert(HasProgramObservers<OutputProgram>);
static_assert(sizeof(IntProgram) == sizeof(std::shared_ptr<void>));
static_assert(sizeof(MultiProgram) == sizeof(std::shared_ptr<void>));
static_assert(sizeof(BoundedProgram) == sizeof(std::shared_ptr<void>));
static_assert(sizeof(OutputProgram) == sizeof(std::shared_ptr<void>));
static_assert(!std::is_copy_constructible_v<IntFlow>);
static_assert(std::is_move_constructible_v<IntFlow>);
static_assert(std::is_copy_constructible_v<IntStage>);
static_assert(std::is_copy_constructible_v<HeterogeneousZip>);
static_assert(std::is_copy_constructible_v<HeterogeneousRecord>);
static_assert(
    std::is_same_v<decltype(std::declval<const IntProgram &>().graph()),
                   const rund::compute::graph::Info &>);
static_assert(noexcept(std::declval<const IntProgram &>().graph()));
static_assert(!AcceptsRvalueFlowInput<std::vector<std::int32_t>>);
static_assert(!AcceptsRvalueFlowInput<std::array<std::int32_t, 4>>);
static_assert(!AcceptsRvalueFlowInput<std::span<std::int32_t>>);
static_assert(AcceptsFlowRange<std::int32_t[4]>);
static_assert(AcceptsConstFlowRange<std::int32_t[4]>);
static_assert(AcceptsFlowRange<SdkContiguousInput<std::int32_t, 4>>);
static_assert(AcceptsConstFlowRange<SdkContiguousInput<std::int32_t, 4>>);
static_assert(AcceptsFlowRange<std::pmr::vector<std::int32_t>>);
static_assert(AcceptsConstFlowRange<std::pmr::vector<std::int32_t>>);
static_assert(!AcceptsFlowRange<std::initializer_list<std::int32_t>>);
static_assert(!AcceptsFlowRange<std::vector<bool>>);
static_assert(!AcceptsFlowRange<std::array<std::int16_t, 4>>);
static_assert(AcceptsGatherRange<std::uint32_t[4]>);
static_assert(AcceptsGatherRange<const std::uint32_t[4]>);
static_assert(AcceptsGatherRange<SdkContiguousInput<std::uint32_t, 4>>);
static_assert(AcceptsGatherRange<const SdkContiguousInput<std::uint32_t, 4>>);
static_assert(!AcceptsGatherRange<std::int32_t[4]>);
static_assert(!AcceptsGatherRange<std::array<std::uint64_t, 4>>);
static_assert(!AcceptsGatherRange<std::vector<bool>>);
static_assert(!AcceptsGatherRange<std::initializer_list<std::uint32_t>>);
static_assert(AcceptsFlowRange<std::vector<std::int32_t>>);
static_assert(AcceptsFlowRange<std::vector<rund::compute::Fixed<1, 31>>>);
static_assert(!AcceptsFlowRange<std::vector<float>>);
static_assert(!AcceptsFlowRange<std::vector<double>>);
static_assert(AcceptsBufferValue<std::uint64_t>);
static_assert(!AcceptsBufferValue<double>);
static_assert(!HasRemainder<rund::compute::Expr<std::int32_t>>);
static_assert(HasFixedDivide<rund::compute::Expr<rund::compute::Fixed<1, 31>>>);
static_assert(!HasFixedDivide<rund::compute::Expr<std::int32_t>>);
static_assert(!HasUnsignedStorageSaturate<rund::compute::Expr<std::int32_t>>);
static_assert(HasUnsignedStorageSaturate<rund::compute::Expr<std::uint32_t>>);
static_assert(HasUnsignedStorageSaturate<
              rund::compute::Expr<rund::compute::Fixed<16, 16>>>);
static_assert(HasUnsignedStorageSaturate<
              rund::compute::Expr<rund::compute::Fixed<20, 44>>>);
static_assert(HasArithmeticShift<rund::compute::Expr<std::int32_t>>);
static_assert(!HasArithmeticShift<rund::compute::Expr<std::uint32_t>>);
static_assert(
    HasArithmeticShift<rund::compute::Expr<rund::compute::Fixed<16, 16>>>);
