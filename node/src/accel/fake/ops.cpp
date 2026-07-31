#include "ops.hpp"
#include "../backend/ops/table.hpp"
#include "../kernel/backend/execute.hpp"

#include <accel/buffer.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include <node/accel/buffer.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickFake(bool allow_fake);

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool allow_fake) {
  return PickFake(allow_fake);
}

[[nodiscard]] rund::Buffer RejectBuffer(const rund::AccelDevice &,
                                        const rund::BufferDesc &,
                                        const BackendBufferInitialization) {
  return rund::Buffer{
      .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
}

[[nodiscard]] rund::AccelCheck RejectTransfer(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &, const void *, std::uint64_t, std::uint64_t) {
  return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
}

[[nodiscard]] BackendDownload RejectDownload(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &, void *, std::uint64_t, std::uint64_t,
    bool) {
  return {};
}

[[nodiscard]] BackendLookup
RejectLookup(const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
             const std::shared_ptr<void> &) {
  return BackendLookup{
      .check = rund::AccelCheck{false, "accel_context_buffer_invalid"}};
}

[[nodiscard]] rund::RuntimeStats Stats(const rund::AccelDevice &) {
  return rund::RuntimeStats{.reason = "accel_buffer_backend_unavailable"};
}

void Reset(const rund::AccelDevice &) {}

[[nodiscard]] rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &) noexcept {
  return {};
}

const BackendOps Operations{
    .api = rund::AccelApi::Fake,
    .create = RejectBuffer,
    .upload = RejectTransfer,
    .download = RejectDownload,
    .lookup = RejectLookup,
    .stats = Stats,
    .reset = Reset,
    .memory = Memory,
    .run = RunFakeKernel,
    .prepare = PrepareFakeKernel,
    .run_batch = nullptr,
    .submit_prepared = SubmitPreparedFakeKernel,
};

} // namespace

BackendEntry FakeEntry() noexcept {
  return BackendEntry{rund::AccelApi::Fake, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
