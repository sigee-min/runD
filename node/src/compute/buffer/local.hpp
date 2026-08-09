#pragma once

#include "../backend.hpp"
#include "../device/state.hpp"

namespace rund::compute::detail {

// These private chunks are never published. A complete graph write or the
// Program's invocation reset precedes every read, so cold allocation does not
// duplicate initialization. Public buffers retain make_buffer's zero contract.
[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_input_binding_buffer(const std::shared_ptr<DeviceState> &device, Type type,
                          std::size_t count);

[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_workspace_buffer(const std::shared_ptr<DeviceState> &device,
                      std::size_t count);

// Source-private physical handoff for callers that already own the public
// WriteStats receipt. The UploadResult is the sole backend-produced transfer
// evidence; CPU writes leave it at its successful zero value.
[[nodiscard]] Status
write_buffer_measured(const std::shared_ptr<BufferState> &buffer,
                      HostView input, WriteStats &stats,
                      UploadResult &transfer) noexcept;

} // namespace rund::compute::detail
