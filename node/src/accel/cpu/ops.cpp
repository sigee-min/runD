#include "ops.hpp"
#include "../backend/buffer.hpp"
#include "../backend/ops/table.hpp"
#include "../backend/usage.hpp"
#include "buffer.hpp"
#include "kernel/run.hpp"
#include "local.hpp"

#include <node/accel/buffer.hpp>

#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickCpu();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickCpu(); }

rund::Buffer Create(const rund::AccelDevice &pick,
                    const rund::BufferDesc &desc,
                    const BackendBufferInitialization) {
  CpuBufferResult created = CreateCpuResidentBuffer(pick, desc);
  return MakeBuffer(pick, desc, created.check, created.ref,
                    std::move(created.buffer), created.ref.bytes, false);
}

rund::AccelCheck Upload(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &ref,
                        const std::shared_ptr<void> &handle, const void *data,
                        const std::uint64_t bytes, const std::uint64_t offset) {
  return UploadCpuResidentBuffer(pick, ref, handle, data, bytes, offset);
}

BackendDownload Download(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle, void *data,
                         const std::uint64_t bytes,
                         const std::uint64_t offset, const bool) {
  return BackendDownload{.check = DownloadCpuResidentBuffer(
                             pick, ref, handle, data, bytes, offset)};
}

BackendLookup Lookup(const rund::AccelDevice &pick,
                     const rund::kernel::ResidentBufferRef &requested,
                     const std::shared_ptr<void> &handle) {
  CpuBufferResult result = LookupCpuResidentBuffer(pick, requested, handle);
  return BackendLookup{.check = result.check,
                       .ref = result.ref,
                       .handle = std::move(result.buffer)};
}

rund::node::accel::AccelMemoryStats Memory(const rund::AccelDevice &) noexcept {
  return {};
}

const BackendOps Operations{
    .api = rund::AccelApi::Cpu,
    .resident = true,
    .create = Create,
    .upload = Upload,
    .download = Download,
    .lookup = Lookup,
    .stats = ReadCpuRuntimeStats,
    .reset = ResetCpuRuntimeStats,
    .memory = Memory,
    .run = RunCpuKernel,
    .prepare = PrepareCpuKernel,
    .run_batch = nullptr,
    .submit_prepared = SubmitPreparedCpuKernel,
};

} // namespace

BackendEntry CpuEntry() noexcept {
  return BackendEntry{rund::AccelApi::Cpu, false, Pick, &Operations};
}

} // namespace rund::node::accel::detail
