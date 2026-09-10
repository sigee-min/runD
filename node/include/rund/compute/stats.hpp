#pragma once

#include <rund/compute/backend.hpp>
#include <rund/compute/pipeline/coordinate.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute {
struct ControlStats final {
  static constexpr std::uint64_t no_overflow =
      std::numeric_limits<std::uint64_t>::max();

  std::uint64_t generated_item_count{};
  std::uint64_t generated_capacity{};
  std::uint64_t indirect_dispatch_count{};
  std::uint64_t indirect_work_item_count{};
  std::uint64_t iteration_count{};
  std::uint64_t skipped_iteration_count{};
  std::uint64_t conflict_count{};
  std::uint64_t overflow_ordinal{no_overflow};

  [[nodiscard]] constexpr bool occupancy_available() const noexcept {
    return generated_capacity != 0u;
  }

  [[nodiscard]] constexpr double occupancy() const noexcept {
    return occupancy_available() ? static_cast<double>(generated_item_count) /
                                       static_cast<double>(generated_capacity)
                                 : 0.0;
  }

  [[nodiscard]] constexpr bool overflowed() const noexcept {
    return overflow_ordinal != no_overflow;
  }

  [[nodiscard]] constexpr bool
  operator==(const ControlStats &) const noexcept = default;
};

struct PublicationStats final {
  std::uint64_t generation{};
  std::uint64_t commit_count{};
  std::uint64_t discard_count{};
  std::uint64_t snapshot_byte_count{};
  std::uint64_t snapshot_hash{};
  std::uint64_t restore_byte_count{};
  std::uint64_t device_loss_count{};

  [[nodiscard]] constexpr bool
  operator==(const PublicationStats &) const noexcept = default;
};

// Physical queue submissions caused by transfer operations are distinct from
// algorithm execution submissions. Keeping the directions explicit lets a
// terminal Profile report one execution submission and one readback submission
// without a benchmark subtracting two snapshots or guessing backend behavior.
struct TransferSubmissionStats final {
  std::uint64_t host_to_device{};
  std::uint64_t device_to_host{};
  std::uint64_t device_to_device{};

  [[nodiscard]] constexpr bool
  operator==(const TransferSubmissionStats &) const noexcept = default;
};

// Pipeline-only checkpoint telemetry is intentionally separate from Stats so
// every Job/Run result does not pay for resident checkpoint state it cannot
// produce.
struct CheckpointStats final {
  std::uint64_t device_state_acquire_count{};
  std::uint64_t device_state_rebase_count{};
  std::uint64_t device_state_copy_byte_count{};
  std::uint64_t device_state_copy_command_count{};
  std::uint64_t reusable_snapshot_count{};
  std::uint64_t reusable_snapshot_byte_count{};
  std::uint64_t reusable_snapshot_hash{};
  std::uint64_t reusable_snapshot_transfer_count{};

  [[nodiscard]] constexpr bool
  operator==(const CheckpointStats &) const noexcept = default;
};

// Pipeline residency is physical execution evidence for a logical dataset
// whose complete byte extent is not retained in the prepared working set.
// It remains nested in Stats so paging cannot create a second public ledger.
struct ResidencyStats final {
  static constexpr std::uint64_t no_failed_page =
      std::numeric_limits<std::uint64_t>::max();

  std::uint64_t logical_bytes{};
  // Exact active element prefix requested by the latest invocation. Capacity
  // and plan identity remain in PipelinePlan::residency.
  std::uint64_t active_count{};
  std::uint64_t page_bytes{};
  std::uint64_t page_count{};
  std::uint64_t frame_capacity{};
  std::uint64_t resident_frames_peak{};
  std::uint64_t epoch_count{};
  // Exact recurrent-controller receipt. These remain zero for CPU, Q=1,
  // rolling, and pre-native fallback. One accepted controller Final is the
  // producer; native batches and queue calls remain backend-owned facts.
  std::uint64_t window_handoff_count{};
  std::uint64_t window_batch_count{};
  std::uint64_t window_queue_call_count{};
  std::uint64_t page_in_count{};
  std::uint64_t page_out_count{};
  std::uint64_t backing_read_bytes{};
  std::uint64_t backing_write_bytes{};
  std::uint64_t backing_io_ns{};
  // Cache and supply evidence is emitted by the same residency authority as
  // page movement. Accelerator page-in bytes are Host-to-Device promotions;
  // they may therefore be nonzero for a reusable Host-tier cache hit. A late
  // page is a demanded backing miss not completed before its execution epoch.
  std::uint64_t cache_hit_count{};
  std::uint64_t eviction_count{};
  std::uint64_t prefetch_count{};
  std::uint64_t late_page_count{};
  std::uint64_t page_in_bytes{};
  std::uint64_t page_out_bytes{};
  std::uint64_t stall_ns{};
  // Directional intersections are recorded from actual transfer receipts and
  // compute execution receipts. overlap_ns is their checked aggregate, never
  // an inference from traffic bytes or submission counts.
  std::uint64_t overlap_ns{};
  std::uint64_t h2d_overlap_ns{};
  std::uint64_t d2h_overlap_ns{};
  std::uint32_t sampled_runs{};
  std::uint32_t allocation_free_runs{};
  std::uint64_t plan_identity_hi{};
  std::uint64_t plan_identity_lo{};
  std::uint64_t failed_page{no_failed_page};

  [[nodiscard]] constexpr bool directional_overlap_exact() const noexcept {
    const std::uint64_t directional =
        h2d_overlap_ns >
                std::numeric_limits<std::uint64_t>::max() - d2h_overlap_ns
            ? std::numeric_limits<std::uint64_t>::max()
            : h2d_overlap_ns + d2h_overlap_ns;
    return overlap_ns == directional;
  }

  [[nodiscard]] constexpr bool
  samples_allocation_free(const std::uint64_t expected) const noexcept {
    return expected < std::numeric_limits<std::uint32_t>::max() &&
           sampled_runs == expected && allocation_free_runs == expected;
  }

  [[nodiscard]] constexpr bool
  operator==(const ResidencyStats &) const noexcept = default;
};

// Preparation counters are meaningful only when their physical producer can
// publish an owner-local receipt. This typed source is part of Stats so a
// Profile serializer never has to infer availability from Backend or from a
// zero-valued counter tuple.
enum class PreparationEvidenceSource : std::uint8_t {
  Unavailable,
  NoNativeProducer,
  OwnerLocal,
  BackendGlobalOnly,
};

struct PipelineStats final {
  static constexpr std::uint64_t no_failed_step =
      std::numeric_limits<std::uint64_t>::max();
  static constexpr std::uint64_t no_coordinate =
      std::numeric_limits<std::uint64_t>::max();

  std::uint64_t step_count{};
  std::uint64_t resource_count{};
  std::uint64_t barrier_count{};
  // Cold-sealed repetitions represented by one physical execution. A value
  // greater than one is admitted only after the exact temporal proof.
  std::uint64_t sealed_repetition_count{};
  // Successful repetitions removed beyond the one physical execution. This
  // remains zero on every failed or unpublished attempt.
  std::uint64_t coalesced_repetition_count{};
  std::uint64_t claim_conflict_count{};
  std::uint64_t verified_step_count{};
  std::uint64_t failed_step_index{no_failed_step};
  std::uint64_t status_entry_count{};
  std::uint64_t control_byte_count{};
  std::uint64_t control_command_count{};
  std::uint64_t executed_outer_window_count{};
  std::uint64_t skipped_outer_window_count{};
  std::uint64_t executed_inner_iteration_count{};
  std::uint64_t skipped_inner_iteration_count{};
  std::uint64_t failed_outer_window{no_coordinate};
  std::uint64_t failed_inner_iteration{no_coordinate};
  PipelineNestedPhase failed_nested_phase{PipelineNestedPhase::None};
  PreparationEvidenceSource preparation_evidence{
      PreparationEvidenceSource::Unavailable};
  std::uint64_t prepared_template_count{};
  std::uint64_t prepared_command_count{};
  // Explicit sample epoch over accepted Pipeline terminals. begin_samples()
  // clears only these counters; ordinary per-run Stats remain the latest
  // execution snapshot. UINT32_MAX is the absorbing saturation value and is
  // never accepted as exact clean-sample evidence.
  std::uint32_t sampled_runs{};
  std::uint32_t clean_runs{};
  std::uint64_t claim_ns{};
  std::uint64_t control_ns{};
  ResidencyStats residency{};

  [[nodiscard]] constexpr bool
  samples_clean(const std::uint64_t expected) const noexcept {
    return expected < std::numeric_limits<std::uint32_t>::max() &&
           sampled_runs == expected && clean_runs == expected;
  }

  [[nodiscard]] constexpr bool
  operator==(const PipelineStats &) const noexcept = default;
};

struct Stats final {
  Backend backend{Backend::Unavailable};
  std::uint64_t pipeline_compiles{};
  std::uint64_t buffer_allocations{};
  std::uint64_t download_events{};
  std::uint64_t dispatches{};
  std::uint64_t command_submits{};
  std::uint64_t command_capacity{};
  std::uint64_t command_inflight_peak{};
  std::uint64_t command_capacity_rejections{};
  std::uint64_t uploaded_bytes{};
  std::uint64_t downloaded_bytes{};
  std::uint64_t pipeline_cache_hits{};
  std::uint64_t pipeline_cache_evictions{};
  std::uint64_t buffer_reuses{};
  std::uint64_t descriptor_pool_creations{};
  std::uint64_t descriptor_set_allocations{};
  std::uint64_t descriptor_reuses{};
  std::uint64_t original_dispatches{};
  std::uint64_t final_dispatches{};
  std::uint64_t fusions{};
  std::uint64_t fusion_rejections{};
  std::uint64_t internal_roundtrip_bytes{};
  std::uint64_t external_roundtrip_bytes{};
  std::uint64_t reset_bytes{};
  std::uint64_t reset_commands{};
  std::uint64_t graph_read_bytes{};
  std::uint64_t kernel_ns{};
  std::uint64_t kernel_samples{};
  std::uint64_t shader_compile_ns{};
  std::uint64_t spirv_compile_ns{};
  std::uint64_t pipeline_create_ns{};
  std::uint64_t descriptor_setup_ns{};
  std::uint64_t submit_wait_ns{};
  std::uint64_t readback_ns{};
  std::uint64_t graph_hash{};
  std::uint64_t output_hash{};
  std::uint32_t worker_count{};
  std::uint32_t participating_workers{};
  std::uint64_t tile_count{};
  std::uint64_t tile_size{};
  std::uint64_t vector_chunks{};
  std::uint64_t tail_chunks{};
  ControlStats control{};
  TransferSubmissionStats transfer_submissions{};
  // HostIteration writes occur after one completed Pipeline execution. Their
  // exact byte total stays in that execution epoch so Profile remains the
  // single observation surface for product measurement. Per-call CPU-copy and
  // accelerator-upload counts remain the existing WriteStats receipt.
  std::uint64_t host_write_bytes{};
  PublicationStats publication{};
  PipelineStats pipeline{};

  [[nodiscard]] constexpr bool kernel_timing_available() const noexcept {
    return kernel_samples != 0u;
  }
  [[nodiscard]] constexpr bool available() const noexcept {
    return backend != Backend::Unavailable;
  }

  [[nodiscard]] constexpr bool
  operator==(const Stats &) const noexcept = default;
};
struct WriteStats final {
  std::uint64_t copies{};
  std::uint64_t uploads{};
  std::uint64_t bytes{};
};
enum class MemoryScope : std::uint8_t {
  Unspecified,
  Program,
  Job,
  Pipeline,
  Backend,
};
enum class MemoryCategory : std::uint8_t {
  Host,
  Frame,
  Tile,
  Resident,
  Staging,
  Device,
  Transfer,
};
enum class MemoryUse : std::uint8_t {
  Metadata,
  Input,
  PendingInput,
  Output,
  Internal,
  Scratch,
  Coordinator,
  Traffic,
};
struct MemoryCounter final {
  std::uint64_t current{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
  std::uint64_t budget{};

  [[nodiscard]] constexpr bool
  operator==(const MemoryCounter &) const noexcept = default;
};
struct MemoryStats final {
  Backend backend{Backend::Unavailable};
  MemoryScope scope{MemoryScope::Unspecified};
  MemoryCounter host{};
  MemoryCounter frame{};
  MemoryCounter tile{};
  MemoryCounter resident{};
  MemoryCounter staging{};
  MemoryCounter device{};
  MemoryCounter transfer{};

  [[nodiscard]] constexpr bool available() const noexcept {
    return backend != Backend::Unavailable && scope != MemoryScope::Unspecified;
  }

  [[nodiscard]] constexpr bool
  operator==(const MemoryStats &) const noexcept = default;
};
struct MemoryEntry final {
  MemoryCategory category{MemoryCategory::Host};
  MemoryUse use{MemoryUse::Metadata};
  std::uint32_t index{};
  MemoryCounter bytes{};
};
struct MemorySnapshot final {
  MemoryStats summary{};
  std::size_t written{};
  std::size_t total{};

  [[nodiscard]] constexpr bool truncated() const noexcept {
    return written < total;
  }
};
} // namespace rund::compute
