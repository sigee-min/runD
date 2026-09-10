#pragma once

#include <rund/compute/device.hpp>

#include <cstdint>
#include <memory>

namespace rund::measure::compute::virtual_residency {

// Measurement-only projection of the production persistent-product Final.
// The observer delegates through the original DeviceOps slots and never
// supplies an alternate lowering or execution result.
struct ProductRouteEvidence final {
  ::rund::compute::Backend backend{::rund::compute::Backend::Unavailable};
  bool production_slots{};
  bool owner_stable{};
  std::uint64_t owner_events{};
  std::uint64_t prepare_attempts{};
  std::uint64_t prepared_runs{};
  std::uint64_t executed_runs{};
  std::uint64_t successful_finals{};
  std::uint64_t cold_owner_runs{};
  std::uint64_t warm_reused_runs{};
  std::uint64_t max_warm_rearm_count{};
  std::uint64_t whole_run_preencoded_runs{};
  std::uint64_t device_generated_recurrence_runs{};
  std::uint64_t fixed_native_storage_runs{};
  std::uint64_t fixed_common_storage_runs{};
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
  // Canonical product-route observations for the retained hash cache. These
  // are first-wins violations collected at each observed execution, not a
  // second cache implementation.
  std::uint64_t hash_observation_changes{};
  std::uint64_t hash_reuse_step_errors{};
  std::uint64_t completed_ns{};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
};

struct ProductRouteObserverImpl;

class ProductRouteObserver final {
public:
  explicit ProductRouteObserver(::rund::compute::Device &) noexcept;
  ProductRouteObserver(const ProductRouteObserver &) = delete;
  ProductRouteObserver &operator=(const ProductRouteObserver &) = delete;
  ~ProductRouteObserver();

  void reset() noexcept;
  [[nodiscard]] const ProductRouteEvidence &evidence() const noexcept;

private:
  std::unique_ptr<ProductRouteObserverImpl> impl_;
};

[[nodiscard]] bool
ExactProductRoute(const ProductRouteEvidence &, ::rund::compute::Backend,
                  std::uint64_t pages, std::uint64_t active_bytes,
                  std::uint64_t expected_runs, bool resident_backing) noexcept;
[[nodiscard]] bool ExactGpuDrivenProductRoute(const ProductRouteEvidence &,
                                              ::rund::compute::Backend,
                                              std::uint64_t pages,
                                              std::uint64_t active_bytes,
                                              std::uint64_t expected_runs,
                                              bool resident_backing) noexcept;
[[nodiscard]] const char *ProductRouteStatus(const ProductRouteEvidence &,
                                             ::rund::compute::Backend,
                                             std::uint64_t pages,
                                             bool resident_backing) noexcept;

} // namespace rund::measure::compute::virtual_residency
