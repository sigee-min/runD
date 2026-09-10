#include "status.hpp"
#include "telemetry.hpp"

#include <memory>
#include <type_traits>

static_assert(HasComputeTelemetry<rund::compute::Stats>);
static_assert(requires(rund::compute::MemoryStats memory) {
  { memory == memory } -> std::same_as<bool>;
});
static_assert(sizeof(rund::compute::ResidencyConfig) == 16u);
static_assert(sizeof(rund::compute::ResidencyPlan) == 64u);
static_assert(sizeof(rund::compute::ResidencyStats) == 232u);
static_assert(sizeof(rund::compute::PipelineStats) == 416u);
static_assert(sizeof(rund::compute::Stats) == 896u);
static_assert(sizeof(rund::compute::MemoryStats) == 288u);
static_assert(sizeof(rund::compute::telemetry::Profile) == 1200u);
static_assert(sizeof(rund::compute::Run) == 1392u);
static_assert(sizeof(rund::compute::Result<rund::compute::Run>) == 1400u);
static_assert(sizeof(rund::compute::PipelinePlan) == 512u);
static_assert(rund::compute::Stats{}.pipeline.preparation_evidence ==
              rund::compute::PreparationEvidenceSource::Unavailable);
static_assert(sizeof(rund::compute::VirtualBacking) == 16u);
static_assert(sizeof(rund::compute::VirtualBuffer<std::int32_t>) ==
              sizeof(std::shared_ptr<void>));
static_assert(
    sizeof(rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>) ==
    sizeof(std::shared_ptr<void>));
static_assert(std::is_abstract_v<rund::compute::VirtualBacking>);
static_assert(
    std::is_copy_constructible_v<rund::compute::VirtualBuffer<std::int32_t>>);
static_assert(!std::is_copy_constructible_v<
              rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>>);
static_assert(std::is_nothrow_move_constructible_v<
              rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>>);

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
