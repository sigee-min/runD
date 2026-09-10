#pragma once

#include "../../model.hpp"
#include "../backing.hpp"
#include "../product_route.hpp"
#include "src/compute/virtual/graph_resident/workload.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include "stats.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rund::measure::compute::virtual_graph_residency {

using Spec = ::rund::compute::graph_resident_workload::Spec;
using Program = Spec::Program;
using Pipeline = ::rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;

struct Facts final {
  bool ok{};
  bool final_received{};
  bool owner_present{};
  bool may_write{};
  std::uintptr_t owner_identity{};
  std::uintptr_t control_identity{};
  bool graph_resident{};
  bool proof_valid{};
  bool alias_reuse{};
  bool output_match{};
  std::uint64_t output_hash{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t proof_digest{};
  std::uint64_t proof_hi{};
  std::uint64_t proof_lo{};
  std::uint64_t generation{};
  std::uint64_t nonce{};
  std::uint32_t resource_count{};
  std::uint32_t owner_count{};
  std::uint32_t endpoint_count{};
  std::uint32_t stage_count{};
  std::uint32_t frame_capacity{};
  std::uint32_t batch_count{};
  std::uint64_t native_submits{};
  std::uint64_t dispatches{};
  std::uint64_t finals{};
  std::uint64_t epoch_submits{};
  std::uint64_t pages{};
  std::uint64_t wavefront_steps{};
  std::uint64_t gpu_read_bytes{};
  std::uint64_t gpu_write_bytes{};
  std::uint64_t host_turns{};
  std::uint64_t host_callbacks{};
  std::uint64_t backing_read_bytes{};
  std::uint64_t backing_write_bytes{};
  std::uint64_t uploaded_bytes{};
  std::uint64_t downloaded_bytes{};
  std::uint64_t internal_roundtrip_bytes{};
  std::uint64_t external_roundtrip_bytes{};
  std::uint64_t generated_pages{};
  std::uint64_t forecasted_pages{};
  std::uint64_t promoted_pages{};
  std::uint64_t completed_pages{};
  std::uint64_t drained_pages{};
  std::uint64_t persisted_pages{};
  std::uint64_t hash_observations{};
  std::uint64_t hash_reuses{};
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
  std::uint64_t topology_digest{Spec::topology_digest()};
  std::uint64_t workload_digest{Spec::workload_digest()};
  std::uint64_t expected_hash{Spec::expected_hash()};
  std::uint64_t input_digest_before{Spec::input_digest()};
  std::uint64_t input_digest_after{};
  double cold_us{};
  Timing timing{};
  Facts cold{};
  virtual_residency::ProductRouteEvidence conditioning{};
  Facts warm{};
  bool owner_stable{};
  bool ok{};
};

[[nodiscard]] bool Run(::rund::compute::Device &device,
                       const ::rund::compute::DeviceInfo &info, Result &result);
[[nodiscard]] bool Validate(const Result &result) noexcept;
[[nodiscard]] bool ReportEnvironment(Backend backend,
                                     const ::rund::compute::Device &device,
                                     const ::rund::compute::DeviceInfo &info);
void Report(const Result &result);

} // namespace rund::measure::compute::virtual_graph_residency
