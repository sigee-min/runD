#include "pipeline.hpp"

#include <type_traits>

static_assert(ConfiguresPipelineProfile<rund::compute::PipelineBuilder>);
static_assert(
    ConfiguresPipelineSealedRepetitions<rund::compute::PipelineBuilder>);
static_assert(rund::compute::PipelineSealedRepetitionCapacity == 1024u);
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
static_assert(rund::compute::Stats{}.pipeline.sealed_repetition_count == 0u);
static_assert(rund::compute::Stats{}.pipeline.coalesced_repetition_count == 0u);
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
