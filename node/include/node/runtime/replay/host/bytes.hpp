#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::replay_detail::payload {

class Store;

// Immutable encoded replay bytes. A producer finishes a mutable vector, then
// freezes its allocation into this shared value. Copies retain one immutable
// owner and expose neither the native allocation nor mutable byte access.
class Bytes final {
public:
  Bytes() noexcept = default;

  [[nodiscard]] static Bytes freeze(std::vector<std::byte> &&bytes) {
    if (bytes.empty()) {
      return {};
    }
    auto owner =
        std::make_shared<const std::vector<std::byte>>(std::move(bytes));
    const std::size_t size = owner->size();
    const std::size_t retained = owner->capacity() * sizeof(std::byte);
    const std::byte *const data = owner->data();
    return Bytes{std::shared_ptr<const void>{std::move(owner)}, data, data,
                 size, size, retained};
  }

  [[nodiscard]] static Bytes
  share(std::shared_ptr<std::vector<std::byte>> owner, const std::size_t offset,
        const std::size_t size) noexcept {
    if (owner == nullptr || offset > owner->size() ||
        size > owner->size() - offset || size == 0u) {
      return {};
    }
    const std::size_t owner_size = owner->size();
    const std::size_t retained = owner->capacity() * sizeof(std::byte);
    const std::byte *const base = owner->data();
    return Bytes{std::shared_ptr<const void>{std::move(owner)}, base,
                 base + offset, owner_size, size, retained};
  }

  // Creates one exact immutable byte owner. The caller may fill `data` only
  // before publishing any copy or slice of the returned value.
  [[nodiscard]] static Bytes create(const std::size_t size, std::byte *&data) {
    if (size == 0u) {
      data = nullptr;
      return {};
    }
    std::shared_ptr<std::byte[]> bytes{new std::byte[size]};
    data = bytes.get();
    return Bytes{std::shared_ptr<const void>{std::move(bytes), data}, data,
                 data, size, size, size};
  }

  [[nodiscard]] Bytes slice(const std::size_t offset,
                            const std::size_t size) const noexcept {
    if (owner_ == nullptr || offset > size_ || size > size_ - offset ||
        size == 0u) {
      return {};
    }
    return Bytes{owner_, owner_data_, data_ + offset, owner_size_, size,
                 retained_bytes_};
  }

  [[nodiscard]] std::span<const std::byte> span() const noexcept {
    return owner_ == nullptr ? std::span<const std::byte>{}
                             : std::span<const std::byte>{data_, size_};
  }

  [[nodiscard]] const std::byte *data() const noexcept {
    return owner_ == nullptr ? nullptr : data_;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return owner_ == nullptr ? 0u : size_;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0u; }

  // Bytes retained by the underlying allocation, not merely by this slice.
  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return owner_ == nullptr ? 0u : retained_bytes_;
  }

private:
  explicit Bytes(std::shared_ptr<const void> owner,
                 const std::byte *const owner_data,
                 const std::byte *const data, const std::size_t owner_size,
                 const std::size_t size,
                 const std::size_t retained_bytes) noexcept
      : owner_(std::move(owner)), owner_data_(owner_data), data_(data),
        owner_size_(owner_size), size_(size), retained_bytes_(retained_bytes) {}

  std::shared_ptr<const void> owner_{};
  const std::byte *owner_data_ = nullptr;
  const std::byte *data_ = nullptr;
  std::size_t owner_size_ = 0u;
  std::size_t size_ = 0u;
  std::size_t retained_bytes_ = 0u;

  friend class Store;
};

} // namespace rund::node::replay_detail::payload
