#pragma once

#include <rund/compute/pipeline/access.hpp>
#include <rund/compute/virtual/backing.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace rund::compute {

template <class Signature> class VirtualPipeline;

namespace detail {
struct VirtualBufferAccess;
} // namespace detail

template <detail::ComputeValue T> class VirtualBuffer final {
public:
  VirtualBuffer(const VirtualBuffer &) noexcept = default;
  VirtualBuffer(VirtualBuffer &&) noexcept = default;
  VirtualBuffer &operator=(const VirtualBuffer &) noexcept = default;
  VirtualBuffer &operator=(VirtualBuffer &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::valid_virtual_buffer(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] std::uint64_t size() const noexcept {
    return detail::virtual_buffer_size(state_);
  }

private:
  template <class> friend class VirtualPipeline;
  friend struct detail::VirtualBufferAccess;
  template <detail::ComputeValue U>
  friend Result<VirtualBuffer<U>>
  virtual_buffer(std::uint64_t, std::shared_ptr<VirtualBacking>) noexcept;
  explicit VirtualBuffer(std::shared_ptr<detail::VirtualBufferState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::VirtualBufferState> state_;
};

namespace detail {

struct VirtualBufferAccess final {
  template <ComputeValue T>
  [[nodiscard]] static const std::shared_ptr<VirtualBufferState> &
  state(const VirtualBuffer<T> &buffer) noexcept {
    return buffer.state_;
  }
};

} // namespace detail

template <detail::ComputeValue T>
[[nodiscard]] Result<VirtualBuffer<T>>
virtual_buffer(const std::uint64_t count,
               std::shared_ptr<VirtualBacking> backing) noexcept {
  auto state = detail::make_virtual_buffer(count, sizeof(T), detail::type<T>(),
                                           detail::storage_format<T>(),
                                           std::move(backing));
  if (!state) {
    return Result<VirtualBuffer<T>>::fail(state.reason());
  }
  return Result<VirtualBuffer<T>>::success(
      VirtualBuffer<T>{std::move(state).value()});
}

// Allocates one logical backing whose byte authority is the exact Buffer
// owned by `device`. The resulting backing remains a normal VirtualBacking:
// callers initialize or observe it through read/write, while an admitted
// DeviceVsm run may bind the resident Buffer directly and omit whole-run Host
// staging. Generic callback backings retain their existing bounded path.
template <detail::ComputeValue T>
[[nodiscard]] Result<std::shared_ptr<VirtualBacking>>
resident_virtual_backing(const Device &device,
                         const std::uint64_t count) noexcept {
  return detail::make_resident_virtual_backing(
      detail::DeviceAccess::state(device), detail::type<T>(), count);
}

} // namespace rund::compute
