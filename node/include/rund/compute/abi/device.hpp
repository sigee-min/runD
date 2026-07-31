#pragma once

#include <rund/compute/abi/model.hpp>
#include <rund/compute/compile.hpp>
#include <rund/compute/target.hpp>
#include <span>
namespace rund::compute::detail {
[[nodiscard]] Result<std::shared_ptr<DeviceState>> open_target(Target target);
[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_target(Target target, Compile resources);
[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_cpu(std::uint32_t workers);
[[nodiscard]] Status own_compile(const std::shared_ptr<DeviceState> &device,
                                 Compile resources);
[[nodiscard]] Status
bind_compile(const std::shared_ptr<DeviceState> &device,
             const std::shared_ptr<CompileService> &service) noexcept;
[[nodiscard]] Compile
device_compile(const std::shared_ptr<DeviceState> &state) noexcept;
[[nodiscard]] Backend
device_backend(const std::shared_ptr<DeviceState> &state) noexcept;
[[nodiscard]] std::size_t
buffer_size(const std::shared_ptr<BufferState> &state) noexcept;
[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_buffer(const std::shared_ptr<DeviceState> &device, Type type,
            std::size_t count);
[[nodiscard]] Result<std::shared_ptr<BufferState>>
upload_raw(const std::shared_ptr<DeviceState> &device, HostView input);
template <class T>
[[nodiscard]] Result<std::shared_ptr<BufferState>>
upload(const std::shared_ptr<DeviceState> &device, std::span<const T> values) {
  return upload_raw(device, HostView{values.data(), values.size(), type<T>()});
}
[[nodiscard]] Status read_typed_raw(const RunState &run,
                                    const std::shared_ptr<BufferState> &buffer,
                                    Type type, void *data, std::size_t bytes,
                                    std::size_t count);
template <class T>
[[nodiscard]] Status read(const RunState &run,
                          const std::shared_ptr<BufferState> &buffer,
                          std::span<T> values) {
  return read_typed_raw(run, buffer, type<T>(), values.data(),
                        values.size_bytes(), values.size());
}
} // namespace rund::compute::detail
