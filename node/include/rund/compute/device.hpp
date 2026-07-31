#pragma once

#include <rund/compute/abi/device.hpp>
#include <rund/compute/abi/observe.hpp>
#include <rund/compute/buffer.hpp>
#include <rund/compute/device/info.hpp>
#include <rund/compute/stats.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund {
class Session;
}

namespace rund::compute {

class FlowBuilder;
class ProgramCache;

namespace detail {
[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_session(::rund::Session &session, Target target);
}

class Device final {
public:
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&) noexcept = default;
  Device &operator=(Device &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] Result<Backend> backend() const noexcept {
    return state_ == nullptr
               ? Result<Backend>::fail(Reason::DeviceInvalid)
               : Result<Backend>::success(detail::device_backend(state_));
  }
  [[nodiscard]] Result<DeviceInfo> info() const noexcept;
  [[nodiscard]] Compile compile() const noexcept {
    return detail::device_compile(state_);
  }
  [[nodiscard]] MemoryStats memory() const noexcept {
    return detail::device_memory(state_);
  }
  [[nodiscard]] MemorySnapshot
  memory_snapshot(const std::span<MemoryEntry> entries) const noexcept {
    return detail::device_memory_snapshot(state_, entries);
  }

  template <detail::ComputeValue T>
  [[nodiscard]] Result<Buffer<T>> buffer(const std::size_t count) const {
    auto result = detail::make_buffer(state_, detail::type<T>(), count);
    if (!result) {
      return Result<Buffer<T>>::fail(result.reason());
    }
    return Result<Buffer<T>>::success(Buffer<T>{std::move(result).value()});
  }

  template <detail::ComputeValue T>
  [[nodiscard]] Result<Buffer<T>>
  upload(const std::span<const T> values) const {
    auto result = detail::upload<T>(state_, values);
    if (!result) {
      return Result<Buffer<T>>::fail(result.reason());
    }
    return Result<Buffer<T>>::success(Buffer<T>{std::move(result).value()});
  }

private:
  friend struct detail::BufferAccess;
  friend struct detail::DeviceAccess;
  friend Result<ProgramCache> program_cache(const Device &, std::size_t);
  friend FlowBuilder on(const Device &, const ProgramCache &) noexcept;
  friend Result<Device> open(Target);
  friend Result<Device> open(Target, Compile);
  friend Result<Device> open(::rund::Session &, Target);
  friend FlowBuilder on(const Device &) noexcept;

  explicit Device(std::shared_ptr<detail::DeviceState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::DeviceState> state_;
};

[[nodiscard]] inline Result<Device> open(const Target target) {
  auto result = detail::open_target(target);
  if (!result) {
    return Result<Device>::fail(result.reason());
  }
  return Result<Device>::success(Device{std::move(result).value()});
}

[[nodiscard]] inline Result<Device> open(const Target target,
                                         const Compile resources) {
  auto result = detail::open_target(target, resources);
  if (!result) {
    return Result<Device>::fail(result.reason());
  }
  return Result<Device>::success(Device{std::move(result).value()});
}

[[nodiscard]] inline Result<Device> open(::rund::Session &session,
                                         const Target target) {
  auto result = detail::open_session(session, target);
  if (!result) {
    return Result<Device>::fail(result.reason());
  }
  return Result<Device>::success(Device{std::move(result).value()});
}

} // namespace rund::compute
