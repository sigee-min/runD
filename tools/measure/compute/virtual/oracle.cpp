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
           preparation.pipeline_compiles == 1u &&
           preparation.pipeline_cache_hits == 1u &&
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
                  const std::uint32_t sampled_runs,
                  const bool gpu_driven_product,
                  const bool resident_backing) noexcept {
  const ::rund::compute::Stats &stats = profile.execution();
  const auto &residency = stats.pipeline.residency;
  const std::uint64_t logical_bytes =
      LogicalElements * sizeof(std::int32_t) * 2u;
  const std::uint64_t page_bytes = PageElements * sizeof(std::int32_t) * 2u;
  const std::uint64_t physical_page_bytes = PageElements * sizeof(std::int32_t);
  const std::uint64_t active_pages =
      active_count / PageElements +
      static_cast<std::uint64_t>(active_count % PageElements != 0u);
  const std::uint64_t active_epochs =
      active_pages / FrameCapacity +
      static_cast<std::uint64_t>(active_pages % FrameCapacity != 0u);
  const std::uint64_t backing_bytes = active_count * sizeof(std::int32_t);
  const std::uint64_t supplied_pages =
      residency.page_in_count + residency.cache_hit_count;
  const bool private_transfer =
      backend == Backend::Vulkan && !gpu_driven_product;
  const std::uint64_t uploaded_bytes =
      private_transfer ? residency.page_in_count * physical_page_bytes : 0u;
  const std::uint64_t downloaded_bytes =
      private_transfer ? active_pages * physical_page_bytes : 0u;
  const std::uint64_t persistent_miss_pages =
      residency.prefetch_count + residency.late_page_count;
  // CPU executes directly in its authoritative Host frames, so every
  // physical-frame miss is also a backing miss. Accelerators may promote a
  // reusable Host-cache hit without another Persistent read.
  const bool exact_tier_supply =
      backend == Backend::Cpu
          ? persistent_miss_pages == residency.page_in_count &&
                residency.page_in_bytes == residency.backing_read_bytes
          : persistent_miss_pages <= residency.page_in_count &&
                residency.backing_read_bytes <= residency.page_in_bytes;
  const bool exact_transfer_submissions =
      gpu_driven_product
          ? stats.transfer_submissions.host_to_device == 0u &&
                stats.transfer_submissions.device_to_host == 0u &&
                stats.transfer_submissions.device_to_device == 0u
      : backend == Backend::Vulkan
          ? (residency.page_in_count == 0u
                 ? stats.transfer_submissions.host_to_device == 0u
                 : stats.transfer_submissions.host_to_device != 0u &&
                       stats.transfer_submissions.host_to_device <=
                           active_epochs) &&
                stats.transfer_submissions.device_to_host == active_epochs
          : stats.transfer_submissions.host_to_device == 0u &&
                stats.transfer_submissions.device_to_host == 0u;
  const bool exact_product_receipt =
      !gpu_driven_product ||
      (residency.window_handoff_count == 1u &&
       residency.window_batch_count == 1u &&
       residency.window_queue_call_count == 1u && stats.command_submits == 1u &&
       stats.command_inflight_peak == 1u);
  const bool resident_bypass = resident_backing && gpu_driven_product;
  const bool exact_backing_bytes =
      resident_bypass ? residency.backing_read_bytes == 0u &&
                            residency.backing_write_bytes == 0u
                      : residency.backing_read_bytes <= backing_bytes &&
                            residency.backing_write_bytes == backing_bytes;
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
         plan.residency.frame_capacity == FrameCapacity &&
         plan.residency.epoch_count == EpochCount &&
         residency.logical_bytes == logical_bytes &&
         residency.active_count == active_count &&
         residency.page_bytes == page_bytes &&
         residency.page_count == PageCount &&
         residency.frame_capacity == FrameCapacity &&
         residency.resident_frames_peak ==
             std::min(active_pages,
                      static_cast<std::uint64_t>(FrameCapacity) * 2u) &&
         residency.epoch_count == active_epochs &&
         supplied_pages == active_pages &&
         residency.page_out_count == active_pages && exact_tier_supply &&
         exact_backing_bytes && residency.page_out_bytes == backing_bytes &&
         residency.plan_identity_hi == plan.residency.identity_hi &&
         residency.plan_identity_lo == plan.residency.identity_lo &&
         residency.sampled_runs == sampled_runs && warm_exact &&
         stats.dispatches == (gpu_driven_product ? 1u : active_pages) &&
         exact_transfer_submissions && exact_product_receipt &&
         stats.uploaded_bytes == uploaded_bytes &&
         stats.downloaded_bytes == downloaded_bytes;
}

} // namespace rund::measure::compute::virtual_residency
