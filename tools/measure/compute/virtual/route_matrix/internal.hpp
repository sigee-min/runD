#pragma once

#include "../../suite/core.hpp"
#include "../backing.hpp"
#include "../model.hpp"

#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rund::compute::detail {
class VirtualDeviceVsmRouteProof;
struct VirtualExecutionDeviceVsmPrepared;
struct VirtualExecutionResult;
struct VirtualPipelineState;
struct VirtualRunProjection;
} // namespace rund::compute::detail

namespace rund::measure::compute::route_matrix {

using Pipeline = ::rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>;
using Backing = ::rund::compute::VirtualBacking;
using ComputeBackend = ::rund::compute::Backend;

struct RouteEvidence final {
  ComputeBackend backend{ComputeBackend::Unavailable};
  bool production_slots{};
  bool owner_stable{};
  bool proof_valid{};
  bool proof_staged_loop{};
  bool proof_graph_resident{};
  bool proof_seen{};
  bool proof_identity_stable{};
  bool proof_flags_stable{};
  std::uint64_t proof_mode{};
  std::uint64_t proof_endpoint{};
  bool capability_consistent{};
  bool per_run_lifecycle_consistent{};
  bool lifecycle_complete{};
  std::uint64_t proof_hi{};
  std::uint64_t proof_lo{};
  std::uint64_t proof_valid_count{};
  std::uint64_t proof_identity_mismatches{};
  std::uint64_t proof_flag_mismatches{};
  std::uint64_t capability_observations{};
  std::uint64_t capability_mismatches{};
  std::uint64_t lifecycle_observations{};
  std::uint64_t lifecycle_mismatches{};
  std::uint64_t owner_events{};
  std::uint64_t prepare_attempts{};
  std::uint64_t prepared_runs{};
  std::uint64_t executed_runs{};
  std::uint64_t successful_finals{};
  std::uint64_t cold_owner_runs{};
  std::uint64_t warm_reused_runs{};
  std::uint64_t max_warm_rearm_count{};
  std::uint64_t device_generated_recurrence_runs{};
  std::uint64_t fixed_native_storage_runs{};
  std::uint64_t fixed_common_storage_runs{};
  std::uint64_t physical_ring_storage_runs{};
  std::uint64_t gpu_addressable_backing_runs{};
  std::uint64_t public_gpu_addressable_backing_runs{};
  std::uint64_t whole_run_staging_runs{};
  std::uint64_t bounded_external_page_service_runs{};
  std::uint64_t one_native_submit_runs{};
  std::uint64_t host_service_turns_zero_runs{};
  std::uint64_t host_epoch_callbacks_zero_runs{};
  std::uint64_t aggregate_terminal_once_runs{};
  std::uint64_t bounded_page_io_runs{};
  std::uint64_t coordinate_count{};
  std::uint64_t accepted_coordinates{};
  std::uint64_t gpu_completed_coordinates{};
  std::uint64_t completed_prefix{};
  std::uint64_t forecasted_pages{};
  std::uint64_t promoted_pages{};
  std::uint64_t drained_pages{};
  std::uint64_t persisted_pages{};
  std::uint64_t overlap_reused_bytes{};
  std::uint64_t gpu_backing_read_bytes{};
  std::uint64_t gpu_backing_write_bytes{};
  std::uint64_t window_seed_dispatches{};
  std::uint64_t window_compute_dispatches{};
  std::uint64_t window_internal_dispatches{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t payload_dispatch_count{};
  std::uint64_t host_service_turn_count{};
  std::uint64_t backend_epoch_callback_count{};
  std::uint64_t host_epoch_callback_count{};
  std::uint64_t backing_wait_count{};
  std::uint64_t backing_signal_count{};
  std::uint64_t backing_acknowledgement_count{};
  std::uint64_t final_callback_count{};
  std::uint64_t queue_calls{};
  std::uint64_t public_handoff_count{};
  std::uint64_t authority_accept_count{};
  std::uint64_t pipeline_terminal_count{};
  std::uint64_t backing_publication_count{};
  std::uint64_t first_output_hash_observation_count{};
  std::uint64_t last_output_hash_observation_count{};
  std::uint64_t first_output_hash_reuse_count{};
  std::uint64_t last_output_hash_reuse_count{};
  std::uint64_t hash_observation_changes{};
  std::uint64_t hash_reuse_step_errors{};
  std::uint64_t completed_ns{};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
};

inline constexpr std::size_t PageElements = 256u;
inline constexpr std::uint32_t FrameCapacity = 2u;
inline constexpr std::size_t WarmSamples = 60u;
inline constexpr std::size_t CohortRuns = 1u + 1u + WarmSamples;
inline constexpr std::size_t PacketCount = 3u;
inline constexpr std::size_t CsvFieldCount = 134u;

// Values are the stable serialized form of the private typed route proof.
inline constexpr std::uint64_t WindowRingMode = 1u;
inline constexpr std::uint64_t ResidentEndpoint = 1u;
inline constexpr std::uint64_t StagedEndpoint = 2u;
inline constexpr std::uint64_t WindowPhysicalCount = 1u;
inline constexpr std::uint64_t WindowLogicalHandoff = 1u;
inline constexpr std::uint64_t WindowLogicalBatch = 1u;
inline constexpr std::uint64_t WindowSeedFactor = 1u;
inline constexpr std::uint64_t WindowComputeFactor = 1u;
inline constexpr std::uint64_t WindowInternalFactor = 2u;

static_assert(CohortRuns == 62u);

enum class ProfileMode : std::uint8_t { Core, Full };

struct CaseSpec final {
  std::string_view family;
  std::string_view backing;
  std::uint64_t q{};
  std::uint64_t window{};
  bool resident{};
  bool supported{};
};

struct ColdTiming final {
  double total_us{};
  double author_us{};
  double compile_us{};
  double prepare_us{};
  double seed_us{};
  double run_us{};
  double read_us{};
};

struct WarmSummary final {
  double p25_us{};
  double p50_us{};
  double p75_us{};
  double p95_us{};
  double mad_us{};
  double elements_per_s{};
  bool complete{};
};

struct RunResult final {
  bool ok{};
  ::rund::compute::Reason reason{::rund::compute::Reason::PipelineInvalid};
  double wall_us{};
};

struct PairEvidence final {
  RunResult cpu;
  RunResult selected;
  bool cpu_output_ok{};
  bool selected_output_ok{};
  std::uint64_t cpu_output_hash{};
  std::uint64_t selected_output_hash{};
  double total_us{};
};

struct WarmCohort final {
  std::array<double, WarmSamples> cpu_values{};
  std::array<double, WarmSamples> selected_values{};
  ::rund::compute::Reason reason{::rund::compute::Reason::PipelineInvalid};
  bool ok{};
};

struct Job final {
  std::unique_ptr<Pipeline> pipeline;
  std::shared_ptr<Backing> input;
  std::shared_ptr<Backing> output;
  ::rund::compute::PipelinePlan plan{};
  ColdTiming cold{};
  bool resident{};
  bool prepared{};
  ::rund::compute::Reason reason{::rund::compute::Reason::PipelineInvalid};
};

struct RouteObserverImpl;

class RouteObserver final {
public:
  explicit RouteObserver(::rund::compute::Device &) noexcept;
  RouteObserver(const RouteObserver &) = delete;
  RouteObserver &operator=(const RouteObserver &) = delete;
  ~RouteObserver();

  void reset() noexcept;
  void finish() noexcept;
  [[nodiscard]] const RouteEvidence &evidence() const noexcept;

private:
  std::unique_ptr<RouteObserverImpl> impl_;
};

struct Row final {
  CaseSpec spec{};
  ProfileMode profile{ProfileMode::Full};
  ComputeBackend backend{ComputeBackend::Unavailable};
  std::string api;
  std::string driver;
  std::string driver_details;
  std::string route;
  std::string expected_route;
  std::string status;
  std::string reason;
  std::string label;
  std::string timing_authority{"unavailable"};
  std::string gpu_kernel_status{"unavailable_no_exact_producer"};
  std::string submit_span_status{"unavailable_no_exact_producer"};
  bool comparable{};
  bool backing_match{true};
  bool output_ok{};
  bool cpu_ok{};
  bool owner_stable{};
  bool one_final{};
  bool allocation_free_60{};
  bool timing_valid{};
  std::uint64_t reason_code{};
  std::uint64_t elements{};
  std::uint64_t page_count{};
  std::uint64_t semantic_hi{};
  std::uint64_t semantic_lo{};
  std::uint64_t plan_hi{};
  std::uint64_t plan_lo{};
  std::uint64_t proof_hi{};
  std::uint64_t proof_lo{};
  std::uint64_t owner_id{};
  std::uint64_t output_hash{};
  std::uint64_t expected_hash{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t gpu_kernel_ns{};
  std::uint64_t submit_span_ns{};
  std::uint64_t cpu_command_submits{};
  std::uint64_t cpu_dispatches{};
  std::uint64_t cpu_output_hash{};
  ColdTiming cold{};
  WarmSummary cpu_warm{};
  WarmSummary backend_warm{};
  ::rund::compute::Stats stats{};
  ::rund::compute::ResidencyStats residency{};
  RouteEvidence route_evidence{};
};

[[nodiscard]] bool parse_backend(std::string_view, ComputeBackend &) noexcept;
[[nodiscard]] bool parse_profile(std::string_view, ProfileMode &) noexcept;
[[nodiscard]] const char *profile_name(ProfileMode) noexcept;

[[nodiscard]] double micros(Clock::duration) noexcept;
[[nodiscard]] std::uint64_t elements(const CaseSpec &) noexcept;
[[nodiscard]] std::uint64_t pages(std::uint64_t) noexcept;
[[nodiscard]] std::uint64_t frame_elements(const CaseSpec &) noexcept;
[[nodiscard]] ::rund::compute::ResidencyConfig
residency_config(const CaseSpec &) noexcept;
[[nodiscard]] Job prepare_job(::rund::compute::Device &, const CaseSpec &, bool,
                              ::rund::compute::ResidencyConfig, bool);
[[nodiscard]] bool fill_tail(const std::shared_ptr<Backing> &,
                             std::uint64_t) noexcept;

[[nodiscard]] RunResult run_one(Job &, const CaseSpec &,
                                std::vector<std::int32_t> *, bool,
                                bool) noexcept;
[[nodiscard]] bool read_output(const Job &, std::uint64_t,
                               std::vector<std::int32_t> &) noexcept;
[[nodiscard]] WarmSummary summarize(std::array<double, WarmSamples> &,
                                    std::uint64_t, bool) noexcept;
[[nodiscard]] WarmCohort run_warm(Job &, Job &, const CaseSpec &, std::uint64_t,
                                  std::uint64_t, std::vector<std::int32_t> &,
                                  std::vector<std::int32_t> &) noexcept;

[[nodiscard]] PairEvidence run_pair(Job &, Job &, const CaseSpec &,
                                    std::uint64_t, std::vector<std::int32_t> &,
                                    std::vector<std::int32_t> &, bool) noexcept;
void hide_row_timing(Row &, const char *) noexcept;
void capture_route(Row &, RouteObserver *) noexcept;
void fail_row(Row &, RouteObserver *, std::string_view, ::rund::compute::Reason,
              const char *, const char * = nullptr);
[[nodiscard]] bool publish_evidence(Row &, Job &, Job &, RouteObserver *,
                                    std::uint64_t, std::uint64_t,
                                    std::vector<std::int32_t> &,
                                    std::vector<std::int32_t> &, WarmCohort &,
                                    const ::rund::compute::Stats &,
                                    const ::rund::compute::Stats &);

[[nodiscard]] bool run_matrix(ComputeBackend, ProfileMode);
[[nodiscard]] bool aggregate();
[[nodiscard]] std::string_view csv_header() noexcept;
void print_header();
void print_row(const Row &);

[[nodiscard]] ::rund::compute::Status prepare_route_observed(
    ::rund::compute::detail::VirtualPipelineState &,
    const ::rund::compute::detail::VirtualRunProjection &,
    ::rund::compute::detail::VirtualDeviceVsmRouteProof,
    ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &) noexcept;
[[nodiscard]] ::rund::compute::detail::VirtualExecutionResult
execute_route_observed(
    ::rund::compute::detail::VirtualPipelineState &,
    std::span<::rund::compute::VirtualBacking *const>,
    ::rund::compute::VirtualBacking &,
    const ::rund::compute::detail::VirtualRunProjection &,
    const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &,
    ::rund::compute::Stats &) noexcept;

namespace oracle {

[[nodiscard]] std::uint64_t hash_values(std::span<const std::int32_t>) noexcept;
[[nodiscard]] std::int32_t seed_value(std::uint64_t) noexcept;
void fill_expected(std::span<std::int32_t>, const CaseSpec &) noexcept;
[[nodiscard]] bool output_ok(std::span<const std::int32_t>, const CaseSpec &,
                             std::uint64_t) noexcept;
[[nodiscard]] const char *route_name(const ::rund::compute::Stats &,
                                     const RouteEvidence &,
                                     std::uint64_t owner_id) noexcept;
[[nodiscard]] const char *mismatch_reason(const Row &) noexcept;
[[nodiscard]] bool exact_route(const Row &) noexcept;
[[nodiscard]] bool cohort_ok(const Row &) noexcept;
[[nodiscard]] bool backing_ok(const Row &) noexcept;
[[nodiscard]] const char *label(const WarmSummary &cpu,
                                const WarmSummary &backend) noexcept;

} // namespace oracle

} // namespace rund::measure::compute::route_matrix
