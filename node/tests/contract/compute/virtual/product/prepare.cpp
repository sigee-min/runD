#include "local.hpp"

#include "backing.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstdint>
#include <limits>
#include <memory>

namespace rund_node_test_virtual::product {
namespace {

[[nodiscard]] bool exact_preparation_evidence(
    const rund::compute::Backend backend,
    const rund::compute::telemetry::Profile &cold,
    const rund::compute::telemetry::Profile &reused) noexcept {
  const rund::compute::Stats &cold_stats = cold.execution();
  const rund::compute::Stats &reused_stats = reused.execution();
  if (!cold.memory().available() || !reused.memory().available() ||
      cold.memory().backend != backend || reused.memory().backend != backend ||
      cold_stats.backend != backend || reused_stats.backend != backend) {
    return false;
  }
  if (backend == rund::compute::Backend::Cpu) {
    return cold_stats.pipeline.preparation_evidence ==
               rund::compute::PreparationEvidenceSource::NoNativeProducer &&
           reused_stats.pipeline.preparation_evidence ==
               rund::compute::PreparationEvidenceSource::NoNativeProducer &&
           cold_stats.pipeline_compiles == 0u &&
           cold_stats.pipeline_cache_hits == 0u &&
           cold_stats.shader_compile_ns == 0u &&
           cold_stats.pipeline_create_ns == 0u &&
           reused_stats.pipeline_compiles == 0u &&
           reused_stats.pipeline_cache_hits == 0u &&
           reused_stats.shader_compile_ns == 0u &&
           reused_stats.pipeline_create_ns == 0u;
  }
  if (backend == rund::compute::Backend::Metal) {
    return cold_stats.pipeline.preparation_evidence ==
               rund::compute::PreparationEvidenceSource::OwnerLocal &&
           reused_stats.pipeline.preparation_evidence ==
               rund::compute::PreparationEvidenceSource::OwnerLocal &&
           cold_stats.pipeline_compiles != 0u &&
           cold_stats.pipeline_cache_hits == 0u &&
           cold_stats.shader_compile_ns != 0u &&
           cold_stats.pipeline_create_ns != 0u &&
           reused_stats.pipeline_compiles == 0u &&
           reused_stats.pipeline_cache_hits != 0u &&
           reused_stats.shader_compile_ns == 0u &&
           reused_stats.pipeline_create_ns == 0u;
  }
  return true;
}

} // namespace

int CheckProductPrepare(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-fused", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return 2;
  }

  auto input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto short_backing = std::make_shared<MemoryVirtualBacking>(
      LogicalBytes - sizeof(std::int32_t), ElementPageBytes);
  auto short_output =
      virtual_buffer<std::int32_t>(LogicalElements - 1u, short_backing);
  if (!input || !output || !short_output) {
    return 3;
  }

  auto same_owner =
      virtual_pipeline(*program, *input, *input, ResidencyConfig{});
  auto same_backing_output =
      virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto same_backing =
      same_backing_output
          ? virtual_pipeline(*program, *input, *same_backing_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  auto wrong_shape =
      virtual_pipeline(*program, *input, *short_output, ResidencyConfig{});
  auto zero_pages = virtual_pipeline(
      *program, *input, *output,
      ResidencyConfig{.device_resident_bytes = 1u, .host_staging_bytes = 1u});
  auto excessive_pages = virtual_pipeline(
      *program, *input, *output,
      ResidencyConfig{
          .device_resident_bytes = std::numeric_limits<std::uint64_t>::max(),
          .host_staging_bytes = std::numeric_limits<std::uint64_t>::max()});
  if (same_owner || same_owner.reason() != Reason::PipelineInvalid ||
      same_backing || same_backing.reason() != Reason::PipelineInvalid ||
      wrong_shape || wrong_shape.reason() != Reason::PrimitiveUnsupported ||
      zero_pages || zero_pages.reason() != Reason::PipelineMemoryBudget ||
      excessive_pages ||
      excessive_pages.reason() != Reason::PipelineMemoryBudget) {
    return 4;
  }

  auto prepared =
      virtual_pipeline(*program, *input, *output, ResidencyConfig{});
  if (!prepared || !*prepared) {
    return 5;
  }
  const PipelinePlan plan = prepared->plan();
  const auto preparation = prepared->profile();
  if (!preparation) {
    return 6;
  }
  const Stats &stats = preparation->execution();
  const ResidencyStats &residency = stats.pipeline.residency;
  if (plan.residency.logical_bytes != LogicalBytes * 2u ||
      plan.residency.logical_bytes <= plan.residency.resident_bytes ||
      plan.residency.page_bytes != ResidencyPageBytes ||
      plan.residency.page_count != PageCount ||
      plan.residency.frame_capacity != FrameCapacity ||
      plan.residency.resident_bytes != FrameBytes * 2u ||
      plan.residency.epoch_count != EpochCount ||
      plan.residency.identity_hi == 0u || plan.residency.identity_lo == 0u ||
      stats.backend != backend ||
      residency.logical_bytes != LogicalBytes * 2u ||
      residency.page_bytes != ResidencyPageBytes ||
      residency.page_count != PageCount ||
      residency.frame_capacity != FrameCapacity ||
      residency.resident_frames_peak != 0u ||
      residency.plan_identity_hi != plan.residency.identity_hi ||
      residency.plan_identity_lo != plan.residency.identity_lo) {
    return 6;
  }

  auto reused_input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto reused_output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto reused_input =
      virtual_buffer<std::int32_t>(LogicalElements, reused_input_backing);
  auto reused_output =
      virtual_buffer<std::int32_t>(LogicalElements, reused_output_backing);
  auto reused = reused_input && reused_output
                    ? virtual_pipeline(*program, *reused_input, *reused_output,
                                       ResidencyConfig{})
                    : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                          Reason::PipelineInvalid);
  auto reused_preparation =
      reused ? reused->profile()
             : Result<telemetry::Profile>::fail(Reason::ProfileInvalid);
  if (!reused || !reused_preparation ||
      !exact_preparation_evidence(backend, *preparation, *reused_preparation)) {
    return 7;
  }
  const BackingFacts input_facts = input_backing->facts();
  const BackingFacts output_facts = output_backing->facts();
  return input_facts.read_count == 0u && input_facts.write_count == 0u &&
                 output_facts.read_count == 0u &&
                 output_facts.write_count == 0u &&
                 input_backing->tail_poisoned() &&
                 output_backing->tail_poisoned()
             ? 0
             : 8;
}

} // namespace rund_node_test_virtual::product
