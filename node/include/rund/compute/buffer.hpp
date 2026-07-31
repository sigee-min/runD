#pragma once

#include <rund/compute/abi/device.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

namespace rund::compute {

class Device;
class Run;
template <class>
class Program;
template <class>
class Buffer;

template <class T> class View final {
public:
  using Element = T;
  using Value = std::remove_const_t<T>;
  View(const View &) noexcept = default;
  View(View &&) noexcept = default;
  View &operator=(const View &) noexcept = default;
  View &operator=(View &&) noexcept = default;

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
  [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
  [[nodiscard]] bool contiguous() const noexcept { return stride_ == 1u; }

  [[nodiscard]] std::uint64_t offset_bytes() const noexcept {
    return static_cast<std::uint64_t>(offset_) * sizeof(Value);
  }
  [[nodiscard]] std::uint64_t stride_bytes() const noexcept {
    return static_cast<std::uint64_t>(stride_) * sizeof(Value);
  }
  [[nodiscard]] std::uint64_t element_bytes() const noexcept {
    return sizeof(Value);
  }

  // Returns the smallest enclosing byte range for diagnostics.  Hazard
  // analysis consumes offset/count/stride separately and therefore never
  // mistakes holes in this envelope for accessed bytes.
  [[nodiscard]] std::uint64_t span_bytes() const noexcept {
    return count_ == 0u
               ? 0u
               : (static_cast<std::uint64_t>(count_ - 1u) * stride_ + 1u) *
                     sizeof(Value);
  }

private:
  template <class> friend class Buffer;
  friend struct detail::BufferAccess;
  View(std::shared_ptr<detail::BufferState> state, const std::size_t offset,
       const std::size_t count, const std::size_t stride,
       const std::size_t alignment) noexcept
      : state_(std::move(state)), offset_(offset), count_(count),
        stride_(stride), alignment_(alignment) {}

  std::shared_ptr<detail::BufferState> state_;
  std::size_t offset_{};
  std::size_t count_{};
  std::size_t stride_{1u};
  std::size_t alignment_{alignof(Value)};
};

template <class T>
class Buffer final {
public:
  Buffer(const Buffer&) noexcept = default;
  Buffer(Buffer&&) noexcept = default;
  Buffer& operator=(const Buffer&) noexcept = default;
  Buffer& operator=(Buffer&&) noexcept = default;

  [[nodiscard]] std::size_t size() const noexcept {
    return detail::buffer_size(state_);
  }

  [[nodiscard]] View<T> view() & noexcept {
    return View<T>{state_, 0u, size(), 1u, alignof(T)};
  }
  [[nodiscard]] View<const T> view() const & noexcept {
    return View<const T>{state_, 0u, size(), 1u, alignof(T)};
  }
  [[nodiscard]] View<T> view() && = delete;
  [[nodiscard]] View<const T> view() const && = delete;

  // Element units keep the surface consistent with Buffer::size().  A view
  // is exact: the last selected element must remain inside this Buffer and
  // every selected element owns sizeof(T) bytes.  Alignment is conservative
  // and derived from the typed base plus offset/stride; it is not a user hint.
  [[nodiscard]] Result<View<T>> view(const std::size_t offset,
                                     const std::size_t count,
                                     const std::size_t stride = 1u) & {
    const std::size_t capacity = size();
    if (stride == 0u) {
      return Result<View<T>>::fail(Reason::BindingInputStrideInvalid);
    }
    const std::size_t canonical_stride = count <= 1u ? 1u : stride;
    if (offset > capacity ||
        (count != 0u &&
         (offset >= capacity ||
          count - 1u > (capacity - 1u - offset) / canonical_stride))) {
      return Result<View<T>>::fail(Reason::ResourceAccessInvalid);
    }
    const std::size_t byte_offset = offset * sizeof(T);
    const std::size_t byte_stride = canonical_stride * sizeof(T);
    const std::size_t alignment =
        std::gcd(alignof(T), std::gcd(byte_offset, byte_stride));
    return Result<View<T>>::success(
        View<T>{state_, offset, count, canonical_stride,
                alignment == 0u ? alignof(T) : alignment});
  }

  [[nodiscard]] Result<View<const T>>
  view(const std::size_t offset, const std::size_t count,
       const std::size_t stride = 1u) const & {
    const std::size_t capacity = size();
    if (stride == 0u) {
      return Result<View<const T>>::fail(Reason::BindingInputStrideInvalid);
    }
    const std::size_t canonical_stride = count <= 1u ? 1u : stride;
    if (offset > capacity ||
        (count != 0u &&
         (offset >= capacity ||
          count - 1u > (capacity - 1u - offset) / canonical_stride))) {
      return Result<View<const T>>::fail(Reason::ResourceAccessInvalid);
    }
    const std::size_t byte_offset = offset * sizeof(T);
    const std::size_t byte_stride = canonical_stride * sizeof(T);
    const std::size_t alignment =
        std::gcd(alignof(T), std::gcd(byte_offset, byte_stride));
    return Result<View<const T>>::success(
        View<const T>{state_, offset, count, canonical_stride,
                      alignment == 0u ? alignof(T) : alignment});
  }
  [[nodiscard]] Result<View<T>> view(std::size_t, std::size_t,
                                     std::size_t = 1u) && = delete;
  [[nodiscard]] Result<View<const T>> view(std::size_t, std::size_t,
                                           std::size_t = 1u) const && = delete;

private:
  friend struct detail::BufferAccess;
  friend class Device;
  friend class Run;
  template <class>
  friend class Program;

  explicit Buffer(std::shared_ptr<detail::BufferState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::BufferState> state_;
};

} // namespace rund::compute
