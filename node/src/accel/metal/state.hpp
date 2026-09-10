#pragma once

#include "adapter.hpp"
#include "../kernel/fault/domain.hpp"
#include "kernel/pipeline/icb.hpp"
#include "stats.hpp"
#include <accel/device.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalPipelineIndex;
struct MetalResidentState;

enum class MetalBufferUsage : std::uint8_t {
  Param,
  Input,
  Output,
  Scratch,
};

struct MetalPipeline {
  rund::kernel::ArtifactKey key{};
  std::uint64_t source_hash{};
  std::string source{};
  std::shared_ptr<void> pipeline{};
};

struct MetalNamedPipeline {
  std::string name{};
  std::shared_ptr<void> pipeline{};
};

struct MetalSourceLibrary {
  std::uint64_t source_hash = 0u;
  std::string source{};
  std::shared_ptr<void> library{};
};

inline constexpr std::size_t kMetalSourceLibraryCapacity = 16u;

struct MetalBuffer {
  std::uint64_t id = 0u;
  rund::kernel::u64 bytes = 0u;
  rund::kernel::u64 allocated_bytes = 0u;
  MetalBufferUsage usage = MetalBufferUsage::Input;
  std::shared_ptr<void> buffer{};
};

struct MetalRuntimeBuffer {
  std::uint64_t id = 0u;
  rund::kernel::u64 bytes = 0u;
  rund::kernel::u64 allocated_bytes = 0u;
  MetalBufferUsage usage = MetalBufferUsage::Input;
  std::shared_ptr<void> buffer{};
  std::uint64_t offset = 0u;
  bool reused = false;
  bool borrowed = false;
};

struct MetalAdapter {
  std::weak_ptr<void> owner_token{};
  std::shared_ptr<void> device{};
  std::shared_ptr<void> queue{};
  // One adapter-owned MTL4 execution queue shared by every prepared residency
  // command owner. Per-Pipeline allocators, command buffers, and exact
  // generation terminals remain independent to preserve bank overlap.
  std::shared_ptr<void> residency_queue{};
  // Immutable device-capability calibration. A successful exact-registry
  // measurement is reused from the process cache and is never charged to an
  // individual Pipeline owner.
  MetalIcbCalibration pipeline_icb_calibration{};
  rund::kernel::ComputeCaps caps{};
  rund::AccelBackendInfo info{};
  std::mutex mutex{};
  // Linearizes native residency acceptance with UnknownMayWrite quarantine.
  // It owns no scheduling policy; Authority has already selected the locals.
  std::mutex residency_terminal_gate{};
  std::condition_variable host_readback_cv{};
  std::size_t active_host_readbacks = 0u;
  std::vector<MetalPipeline> pipelines{};
  std::unique_ptr<MetalPipelineIndex> pipeline_index{};
  std::vector<MetalNamedPipeline> named_pipelines{};
  std::vector<MetalSourceLibrary> source_libraries{};
  std::vector<MetalBuffer> free_buffers{};
  std::unique_ptr<MetalResidentState> resident{};
  std::uint64_t next_runtime_buffer_id = 1u;
  MetalRuntimeStats stats{};
  MetalMemoryStats memory{};
  const char *last_error = "ok";
  std::array<char, 1024u> last_error_detail{};
  std::atomic<bool> fault_source_library_publish_once{false};
  std::atomic<bool> fault_named_pipeline_publish_once{false};
  DeviceLossFault device_loss_fault{};
  std::atomic<bool> fault_download_once{false};
  mutable std::atomic<bool> fault_host_read_once{false};
  mutable std::atomic<bool> fault_host_write_once{false};
  std::atomic<bool> fault_residency_terminal_once{false};
  std::atomic<bool> residency_quarantined{false};
  std::atomic<std::uint64_t> residency_command_active{};
  std::atomic<std::uint64_t> residency_command_peak{};
  std::atomic<bool> fault_trace_unavailable_once{false};
  std::atomic<bool> fault_trace_resolve_device_lost_once{false};

  MetalAdapter();
  MetalAdapter(const MetalAdapter &) = delete;
  MetalAdapter &operator=(const MetalAdapter &) = delete;
  ~MetalAdapter();
};

void SetMetalLastError(MetalAdapter &adapter, const char *reason) noexcept;
void SetMetalLastErrorDetail(MetalAdapter &adapter,
                             const char *reason) noexcept;
[[nodiscard]] const char *MetalLastError(void *context) noexcept;

} // namespace rund::node::accel::detail
