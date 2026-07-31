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
};

struct PublicationStats final {
  std::uint64_t generation{};
  std::uint64_t commit_count{};
  std::uint64_t discard_count{};
  std::uint64_t snapshot_byte_count{};
  std::uint64_t snapshot_hash{};
  std::uint64_t restore_byte_count{};
  std::uint64_t device_loss_count{};
};

struct PipelineStats final {
  static constexpr std::uint64_t no_failed_step =
      std::numeric_limits<std::uint64_t>::max();
  static constexpr std::uint64_t no_coordinate =
      std::numeric_limits<std::uint64_t>::max();

  std::uint64_t step_count{};
  std::uint64_t resource_count{};
  std::uint64_t barrier_count{};
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
  std::uint64_t prepared_template_count{};
  std::uint64_t prepared_command_count{};
  // Post-prepare mutations of retained Job/Buffer/View binding identity.
  // Encoding an immutable descriptor into a fresh native command buffer is
  // not a mutation. The current prepared Pipeline has no warm mutation path,
  // so this is zero by construction; structural contract tests independently
  // compare the frozen owners and descriptors across executions.
  std::uint64_t rebinding_count{};
  std::uint64_t claim_ns{};
  std::uint64_t control_ns{};
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
  PublicationStats publication{};
  PipelineStats pipeline{};

  [[nodiscard]] constexpr bool kernel_timing_available() const noexcept {
    return kernel_samples != 0u;
  }
  [[nodiscard]] constexpr bool available() const noexcept {
    return backend != Backend::Unavailable;
  }
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
