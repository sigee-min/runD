#pragma once

#include "adapter.hpp"
#include "stats.hpp"
#include <accel/device.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

#include <condition_variable>
#include <atomic>
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
  MetalBufferUsage usage = MetalBufferUsage::Input;
  std::shared_ptr<void> buffer{};
};

struct MetalRuntimeBuffer {
  std::uint64_t id = 0u;
  rund::kernel::u64 bytes = 0u;
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
  rund::kernel::ComputeCaps caps{};
  rund::AccelBackendInfo info{};
  std::mutex mutex{};
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
  std::atomic<bool> fault_device_lost_once{false};

  MetalAdapter();
  MetalAdapter(const MetalAdapter &) = delete;
  MetalAdapter &operator=(const MetalAdapter &) = delete;
  ~MetalAdapter();
};

void SetMetalLastError(MetalAdapter &adapter, const char *reason) noexcept;
[[nodiscard]] const char *MetalLastError(void *context) noexcept;

} // namespace rund::node::accel::detail
