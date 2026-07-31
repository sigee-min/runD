#pragma once

#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../resident/model.hpp"
#include "local.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

struct CpuBuffer : ResidentEntry {
  std::vector<std::uint8_t> data{};
};

struct CpuBufferResult {
  rund::AccelCheck check{};
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<CpuBuffer> buffer{};
};

[[nodiscard]] bool CpuPickOwnsAdapter(const rund::AccelDevice &pick) noexcept;
[[nodiscard]] CpuAdapter *
CpuAdapterFromPick(const rund::AccelDevice &pick) noexcept;
[[nodiscard]] CpuBufferResult
CreateCpuResidentBuffer(const rund::AccelDevice &pick,
                        const rund::BufferDesc &desc);
[[nodiscard]] CpuBufferResult
LookupCpuResidentBuffer(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &requested,
                        const std::shared_ptr<void> &handle);
[[nodiscard]] CpuBufferResult
LookupCpuResidentView(const rund::AccelDevice &pick,
                      const rund::kernel::ResidentBufferRef &requested,
                      const std::shared_ptr<void> &handle, std::uint32_t usage);
[[nodiscard]] rund::AccelCheck
UploadCpuResidentBuffer(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &requested,
                        const std::shared_ptr<void> &handle, const void *data,
                        std::uint64_t bytes, std::uint64_t offset);
[[nodiscard]] rund::AccelCheck
DownloadCpuResidentBuffer(const rund::AccelDevice &pick,
                          const rund::kernel::ResidentBufferRef &requested,
                          const std::shared_ptr<void> &handle, void *data,
                          std::uint64_t bytes, std::uint64_t offset);
[[nodiscard]] rund::RuntimeStats
ReadCpuRuntimeStats(const rund::AccelDevice &pick);
void ResetCpuRuntimeStats(const rund::AccelDevice &pick);
void RecordCpuDispatches(CpuAdapter &adapter, std::uint64_t count);

} // namespace rund::node::accel::detail
