#include "oracle.hpp"

#include "model.hpp"

#include <rund/compute/stats.hpp>

#include <algorithm>
#include <bit>

namespace rund::measure::compute::virtual_residency {
namespace {

[[nodiscard]] std::int32_t seed_value(const std::size_t index) noexcept {
  return static_cast<std::int32_t>(index % 4'093u) - 2'046;
}

[[nodiscard]] std::int32_t expected_value(const std::int32_t value) noexcept {
  return (value + 5) * 3;
}

} // namespace

void Seed(const std::span<std::int32_t> values) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = seed_value(index);
  }
}

bool ValidOutput(const std::span<const std::int32_t> values,
                 const std::size_t active_count) noexcept {
  if (values.size() != LogicalElements) {
    return false;
  }
  for (std::size_t index = 0u; index < active_count; ++index) {
    if (values[index] != expected_value(seed_value(index))) {
      return false;
    }
  }
  const auto bytes = std::as_bytes(values);
  const std::size_t active_bytes = active_count * sizeof(std::int32_t);
  return std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(active_bytes),
                     bytes.end(),
                     [](const std::byte value) { return value == TailPoison; });
}

std::uint64_t ContentHash(const std::span<const std::int32_t> values,
                          const std::size_t active_count) noexcept {
  constexpr std::uint64_t offset = 1'469'598'103'934'665'603ull;
  constexpr std::uint64_t prime = 1'099'511'628'211ull;
  std::uint64_t hash = offset;
  for (const std::int32_t value : values.first(active_count)) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (std::uint32_t byte = 0u; byte < sizeof(bits); ++byte) {
      hash ^= (bits >> (byte * 8u)) & 0xffu;
      hash *= prime;
    }
  }
  return hash;
}

bool ExactPreparationEvidence(
    const ::rund::compute::telemetry::Profile &profile) noexcept {
  const ::rund::compute::Stats &preparation = profile.execution();
  const bool numeric_zero = preparation.pipeline_compiles == 0u &&
                            preparation.pipeline_cache_hits == 0u &&
                            preparation.pipeline_cache_evictions == 0u &&
                            preparation.buffer_allocations == 0u &&
                            preparation.buffer_reuses == 0u &&
                            preparation.descriptor_pool_creations == 0u &&
                            preparation.descriptor_set_allocations == 0u &&
                            preparation.descriptor_reuses == 0u &&
                            preparation.shader_compile_ns == 0u &&
                            preparation.spirv_compile_ns == 0u &&
                            preparation.pipeline_create_ns == 0u &&
                            preparation.descriptor_setup_ns == 0u;
  switch (preparation.pipeline.preparation_evidence) {
  case ::rund::compute::PreparationEvidenceSource::OwnerLocal:
    return profile.memory().available() &&
           profile.memory().backend == preparation.backend &&
           preparation.pipeline_compiles != 0u &&
           preparation.pipeline_cache_hits == 0u &&
           preparation.pipeline_cache_evictions == 0u &&
           preparation.shader_compile_ns != 0u &&
           preparation.pipeline_create_ns != 0u;
  case ::rund::compute::PreparationEvidenceSource::NoNativeProducer:
  case ::rund::compute::PreparationEvidenceSource::BackendGlobalOnly:
    return profile.memory().available() &&
           profile.memory().backend == preparation.backend && numeric_zero;
  case ::rund::compute::PreparationEvidenceSource::Unavailable:
    return false;
  }
  return false;
}

bool ExactProfile(const ::rund::compute::telemetry::Profile &profile,
                  const ::rund::compute::PipelinePlan &plan,
                  const Backend backend, const std::size_t active_count,
                  const std::uint32_t sampled_runs) noexcept {
  const ::rund::compute::Stats &stats = profile.execution();
  const auto &residency = stats.pipeline.residency;
  const std::uint64_t logical_bytes =
      LogicalElements * sizeof(std::int32_t) * 2u;
  const std::uint64_t page_bytes = PageElements * sizeof(std::int32_t) * 2u;
  const std::uint64_t physical_page_bytes = PageElements * sizeof(std::int32_t);
  const std::uint64_t active_pages =
      active_count / PageElements +
      static_cast<std::uint64_t>(active_count % PageElements != 0u);
  const std::uint64_t active_waves =
      active_pages / SlotCapacity +
      static_cast<std::uint64_t>(active_pages % SlotCapacity != 0u);
  const std::uint64_t backing_bytes = active_count * sizeof(std::int32_t);
  const std::uint64_t transfer_bytes =
      active_waves * SlotCapacity * physical_page_bytes;
  const bool warm_exact =
      sampled_runs == 0u ||
      (residency.samples_allocation_free(sampled_runs) &&
       stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
       stats.descriptor_pool_creations == 0u &&
       stats.descriptor_set_allocations == 0u);

  return stats.backend == backend && profile.memory().available() &&
         profile.memory().backend == backend && stats.output_hash != 0u &&
         plan.residency.logical_bytes == logical_bytes &&
         plan.residency.page_bytes == page_bytes &&
         plan.residency.page_count == PageCount &&
         plan.residency.slot_capacity == SlotCapacity &&
         plan.residency.wave_count == WaveCount &&
         residency.logical_bytes == logical_bytes &&
         residency.active_count == active_count &&
         residency.page_bytes == page_bytes &&
         residency.page_count == PageCount &&
         residency.slot_capacity == SlotCapacity &&
         residency.active_slots_peak ==
             std::min(active_pages, static_cast<std::uint64_t>(SlotCapacity)) &&
         residency.wave_count == active_waves &&
         residency.load_count == active_pages &&
         residency.writeback_count == active_pages &&
         residency.backing_read_bytes == backing_bytes &&
         residency.backing_write_bytes == backing_bytes &&
         residency.plan_identity_hi == plan.residency.identity_hi &&
         residency.plan_identity_lo == plan.residency.identity_lo &&
         residency.sampled_runs == sampled_runs && warm_exact &&
         stats.dispatches == active_waves * SlotCapacity &&
         stats.transfer_submissions.host_to_device == 0u &&
         stats.transfer_submissions.device_to_host == 0u &&
         stats.uploaded_bytes == transfer_bytes &&
         stats.downloaded_bytes == transfer_bytes;
}

} // namespace rund::measure::compute::virtual_residency
