#include "model.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"

#include <rund/compute/pipeline/runtime.hpp>

#include <span>

namespace rund::compute::detail::virtual_run_overlap {

bool cycle_flight(const PreparedEpoch &prepared,
                  const VirtualRunProjection &run,
                  residency::cycle::Flight &flight) noexcept {
  if (prepared.pipeline == nullptr || prepared.pipeline->device == nullptr ||
      prepared.token == 0u || prepared.binding_count == 0u ||
      prepared.binding_count > 32u || run.frame_capacity == 0u ||
      run.frame_capacity > 32u) {
    return false;
  }
  std::uint32_t input_mask = 0u;
  std::uint32_t output_mask = 0u;
  const residency::FrameRegion input =
      prepared.pipeline->residency_pool
          ->input_regions[prepared.pipeline->residency_bank];
  const std::uint32_t output_first =
      prepared.pipeline->residency_pool->first_output_frame +
      prepared.pipeline->residency_bank *
          static_cast<std::uint32_t>(run.frame_capacity);
  for (std::size_t index = 0u; index < prepared.binding_count; ++index) {
    const residency::CacheBinding source = prepared.bindings[index];
    const residency::CacheBinding target =
        prepared.bindings[prepared.binding_count + index];
    if (source.frame < input.first ||
        source.frame - input.first >= input.count ||
        target.frame < output_first ||
        target.frame - output_first >= run.frame_capacity) {
      return false;
    }
    input_mask |= std::uint32_t{1u} << (source.frame - input.first);
    output_mask |= std::uint32_t{1u} << (target.frame - output_first);
  }
  const residency::cycle::Domain domain =
      prepared.pipeline->device->backend == Backend::Metal
          ? residency::cycle::Domain::HostVisible
          : residency::cycle::Domain::Device;
  flight = residency::cycle::Flight{
      .ordinal = prepared.ordinal,
      .token = prepared.token,
      .bank = prepared.pipeline->residency_bank,
      .input = residency::cycle::Epoch::Range{
          .domain = domain,
          .first = input.first,
          .count = input.count,
          .mask = input_mask,
      },
      .output = residency::cycle::Epoch::Range{
          .domain = domain,
          .first = output_first,
          .count = static_cast<std::uint32_t>(run.frame_capacity),
          .mask = output_mask,
      },
      .may_write = true,
  };
  return true;
}

residency::EpochLease input_lease(PreparedEpoch &prepared) noexcept {
  return residency::EpochLease{
      .bindings = std::span<const residency::CacheBinding>{
          prepared.bindings.data(), prepared.binding_count},
      .transitions = std::span<const residency::CacheTransition>{
          prepared.transitions.data(), prepared.transition_count},
      .token = prepared.token,
  };
}

residency::EpochLease
execution_output_lease(PreparedEpoch &prepared) noexcept {
  return residency::EpochLease{
      .bindings = std::span<const residency::CacheBinding>{
          prepared.bindings.data() + prepared.binding_count,
          prepared.binding_count},
      .token = prepared.token,
  };
}

residency::EpochLease execution_lease(PreparedEpoch &prepared) noexcept {
  return residency::EpochLease{
      .bindings = std::span<const residency::CacheBinding>{
          prepared.bindings.data(), prepared.binding_count * 2u},
      .transitions = std::span<const residency::CacheTransition>{
          prepared.transitions.data(), prepared.transition_count},
      .token = prepared.token,
  };
}

Status fold_epoch(PreparedEpoch &prepared, Stats &stats) noexcept {
  if (prepared.stats_folded) {
    return Status::success();
  }
  if (prepared.pipeline == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const Status folded =
      accumulate_virtual_epoch(stats, pipeline_stats(prepared.pipeline));
  if (folded) {
    prepared.stats_folded = true;
  }
  return folded;
}

} // namespace rund::compute::detail::virtual_run_overlap
