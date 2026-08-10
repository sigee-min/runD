#pragma once

#include "../backend.hpp"
#include "../device/state.hpp"

namespace rund::compute::detail {

// Projects the exact retained charge that the backend will publish through
// BufferState::physical_bytes before Pipeline materialization. CPU storage has
// no backend allocation padding; accelerator backends own their requirement.
[[nodiscard]] inline Result<std::uint64_t>
planned_buffer_storage_bytes(const DeviceState &device,
                             const std::uint64_t logical_bytes) noexcept {
  if (logical_bytes == 0u) {
    return Result<std::uint64_t>::success(0u);
  }
  if (device.backend == Backend::Cpu) {
    return Result<std::uint64_t>::success(logical_bytes);
  }
  if (device.ops == nullptr || device.ops->buffer_storage_bytes == nullptr) {
    return Result<std::uint64_t>::fail(Reason::DeviceInvalid);
  }
  const std::uint64_t committed =
      device.ops->buffer_storage_bytes(device, logical_bytes);
  return committed < logical_bytes
             ? Result<std::uint64_t>::fail(Reason::BufferCapacity)
             : Result<std::uint64_t>::success(committed);
}

// These private chunks are never published. A complete graph write or the
// Program's invocation reset precedes every read, so cold allocation does not
// duplicate initialization. Public buffers retain make_buffer's zero contract.
[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_input_binding_buffer(const std::shared_ptr<DeviceState> &device, Type type,
                          std::size_t count);

[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_planned_input_binding_buffer(const std::shared_ptr<DeviceState> &device,
                                  Type type, std::size_t count,
                                  std::uint64_t exact_storage_bytes);

[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_workspace_buffer(const std::shared_ptr<DeviceState> &device,
                      std::size_t count);

[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_planned_workspace_buffer(const std::shared_ptr<DeviceState> &device,
                              std::size_t count);

// Source-private physical handoff for callers that already own the public
// WriteStats receipt. The UploadResult is the sole backend-produced transfer
// evidence; CPU writes leave it at its successful zero value.
[[nodiscard]] Status
write_buffer_measured(const std::shared_ptr<BufferState> &buffer,
                      HostView input, WriteStats &stats,
                      UploadResult &transfer) noexcept;

} // namespace rund::compute::detail
