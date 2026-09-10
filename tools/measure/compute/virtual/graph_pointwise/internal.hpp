#pragma once

#include "../backing.hpp"
#include "../graph_pointwise.hpp"
#include "../graph_residency/stats.hpp"
#include "../product_route.hpp"

#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace rund::measure::compute::virtual_graph_pointwise {

using Pipeline = ::rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;
using Timing = virtual_graph_residency::Timing;
inline constexpr std::size_t SampleCount = virtual_graph_residency::SampleCount;

struct Facts final {
  bool ok{};
  bool status_observed{};
  ::rund::compute::Code status_code{::rund::compute::Code::Invalid};
  ::rund::compute::Reason status_reason{::rund::compute::Reason::ReasonInvalid};
  std::string status_error;
  bool failure_present{};
  std::uint8_t failure_phase{};
  std::uint8_t failure_check{};
  std::uint32_t failure_stage{std::numeric_limits<std::uint32_t>::max()};
  std::uint64_t failure_batch{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t failure_page{::rund::compute::ResidencyStats::no_failed_page};
  bool cpu_reference{};
  bool final_received{};
  bool owner_present{};
  bool may_write{};
  bool graph_pointwise{};
  bool proof_valid{};
  bool page_map_valid{};
  bool output_match{};
  std::uintptr_t owner_identity{};
  std::uintptr_t control_identity{};
  std::uint64_t graph_hi{};
  std::uint64_t graph_lo{};
  std::uint64_t output_hash{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t proof_digest{};
  std::uint64_t proof_hi{};
  std::uint64_t proof_lo{};
  std::uint64_t generation{};
  std::uint64_t nonce{};
  std::uint32_t input_count{};
  std::uint32_t stage_count{};
  std::uint32_t page_count{};
  std::uint32_t tail_elements{};
  std::uint32_t frame_capacity{};
  std::uint32_t batch_count{};
  std::uint32_t map_rows{};
  std::uint32_t workgroup_width{};
  std::uint64_t native_submits{};
  std::uint64_t dispatches{};
  std::uint64_t finals{};
  std::uint64_t epoch_submits{};
  std::uint64_t generated_pages{};
  std::uint64_t forecasted_pages{};
  std::uint64_t promoted_pages{};
  std::uint64_t completed_pages{};
  std::uint64_t drained_pages{};
  std::uint64_t persisted_pages{};
  std::uint64_t host_turns{};
  std::uint64_t host_callbacks{};
  std::uint64_t gpu_read_bytes{};
  std::uint64_t gpu_write_bytes{};
  std::uint64_t uploaded_bytes{};
  std::uint64_t downloaded_bytes{};
  std::uint64_t backing_read_bytes{};
  std::uint64_t backing_write_bytes{};
  ::rund::compute::Stats stats{};
  ::rund::compute::ResidencyStats residency{};
  bool quarantined{};
  virtual_residency::ProductRouteEvidence route{};
};

struct Result final {
  Backend backend{Backend::Unavailable};
  bool device_valid{};
  std::uint32_t device_code{};
  std::string device_error;
  std::string driver;
  std::string driver_details;
  std::string device;
  std::uint64_t graph_hi{};
  std::uint64_t graph_lo{};
  std::uint64_t expected_hash{Spec::expected_hash()};
  std::uint64_t input_digest_before{};
  std::uint64_t input_digest_after{};
  std::uint64_t completed_samples{};
  bool timing_complete{};
  double cold_us{};
  Timing timing{};
  Facts cold{};
  virtual_residency::ProductRouteEvidence conditioning{};
  Facts warm{};
  bool owner_stable{};
  bool ok{};
};

struct OracleDiagnostic final {
  std::uint64_t failed_mask{};
  std::uint32_t first_leaf{std::numeric_limits<std::uint32_t>::max()};
  const char *first_leaf_name{};

  [[nodiscard]] constexpr bool clean() const noexcept {
    return failed_mask == 0u;
  }
};

[[nodiscard]] OracleDiagnostic Diagnose(const Result &) noexcept;
[[nodiscard]] bool Validate(const Result &result) noexcept;
[[nodiscard]] bool Run(::rund::compute::Device &device,
                       const ::rund::compute::DeviceInfo &info, Result &result);
[[nodiscard]] bool ReportEnvironment(Backend backend,
                                     const ::rund::compute::Device &device,
                                     const ::rund::compute::DeviceInfo &info);
void Report(const Result &result);

} // namespace rund::measure::compute::virtual_graph_pointwise

namespace rund::measure::compute {
void PrintVirtualGraphPointwiseColumns();
} // namespace rund::measure::compute
