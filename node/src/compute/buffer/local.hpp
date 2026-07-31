#pragma once

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

} // namespace rund::compute::detail
