#pragma once

#include "../../kernel/callback.hpp"
#include "../result.hpp"

#include <accel/api.hpp>
#include <accel/check.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rund {
struct AccelDevice;
struct Buffer;
struct BufferDesc;
struct RuntimeStats;
} // namespace rund

namespace rund::node::accel {
struct AccelMemoryStats;
} // namespace rund::node::accel

namespace rund::node::accel::detail {

struct BackendRun;
struct BackendBatchEntry;
struct BackendPublish;
struct PreparedMemory;
struct PreparedPipelineMemory;
class PreparedMemoryMeter;
struct PreparedPipelineStatusLayout;

enum class BackendBufferInitialization : std::uint8_t {
  Zeroed,
  FullOverwrite,
};

struct BackendOps final {
  rund::AccelApi api = rund::AccelApi::Auto;
  bool resident = false;
  rund::Buffer (*create)(const rund::AccelDevice &, const rund::BufferDesc &,
                         BackendBufferInitialization) = nullptr;
  rund::AccelCheck (*upload)(const rund::AccelDevice &,
                             const rund::kernel::ResidentBufferRef &,
                             const std::shared_ptr<void> &, const void *,
                             std::uint64_t, std::uint64_t) = nullptr;
  BackendUpload (*upload_batch)(const rund::AccelDevice &,
                                std::span<const UploadRoute>) =
      nullptr;
  BackendDownload (*download)(const rund::AccelDevice &,
                              const rund::kernel::ResidentBufferRef &,
                              const std::shared_ptr<void> &, void *,
                              std::uint64_t, std::uint64_t, bool) = nullptr;
  BackendDownload (*download_batch)(const rund::AccelDevice &,
                                    std::span<const DownloadRoute>) =
      nullptr;
  BackendLookup (*lookup)(const rund::AccelDevice &,
                          const rund::kernel::ResidentBufferRef &,
                          const std::shared_ptr<void> &) = nullptr;
  rund::RuntimeStats (*stats)(const rund::AccelDevice &) = nullptr;
  void (*reset)(const rund::AccelDevice &) = nullptr;
  rund::node::accel::AccelMemoryStats (*memory)(
      const rund::AccelDevice &) noexcept = nullptr;
  rund::AccelCheck (*run)(const BackendRun &) = nullptr;
  rund::AccelCheck (*prepare)(const BackendRun &, std::shared_ptr<void> &,
                              PreparedMemory &) = nullptr;
  rund::AccelCheck (*prepare_pipeline_private)(const BackendRun &,
                                               std::shared_ptr<void> &,
                                               PreparedMemory &) = nullptr;
  std::uint64_t (*traffic)(const std::shared_ptr<void> &) noexcept = nullptr;
  rund::AccelCheck (*run_batch)(std::span<const BackendBatchEntry>,
                                std::span<rund::AccelCheck>,
                                std::shared_ptr<void> &,
                                rund::RuntimeStats &) = nullptr;
  rund::AccelCheck (*prepare_pipeline)(std::span<const BackendBatchEntry>,
                                       std::span<const BackendBatchEntry>,
                                       std::span<const std::uint8_t>,
                                       std::span<const BackendPublish>,
                                       PreparedPipelineStatusLayout &, bool,
                                       std::shared_ptr<void> &,
                                       PreparedPipelineMemory &) = nullptr;
  rund::AccelCheck (*seed_prepared_pipeline_generation)(
      const std::shared_ptr<void> &, std::uint32_t) noexcept = nullptr;
  rund::AccelCheck (*submit_prepared_pipeline)(const std::shared_ptr<void> &,
                                               KernelCompletion,
                                               void *) noexcept = nullptr;
  rund::AccelCheck (*submit_prepared)(
      const BackendRun &, const std::shared_ptr<void> &, KernelCompletion,
      void *, PreparedMemoryMeter *,
      const std::shared_ptr<void> &) noexcept = nullptr;
  bool (*inject_device_lost_once)(const rund::AccelDevice &) noexcept = nullptr;
};

} // namespace rund::node::accel::detail
