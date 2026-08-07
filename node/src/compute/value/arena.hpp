#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute::detail {

struct ValueIdRange final {
  std::uint32_t offset{};
  std::uint32_t count{};
};

struct ValueRoutes final {
  ValueIdRange inputs{};
  ValueIdRange outputs{};
};

static_assert(std::is_trivially_copyable_v<ValueIdRange>);
static_assert(std::is_trivially_copyable_v<ValueRoutes>);
static_assert(sizeof(ValueIdRange) == 2u * sizeof(std::uint32_t));
static_assert(sizeof(ValueRoutes) == 4u * sizeof(std::uint32_t));

class ValueIdArena final {
public:
  ValueIdArena() = default;
  ValueIdArena(const ValueIdArena &) = delete;
  ValueIdArena &operator=(const ValueIdArena &) = delete;
  ValueIdArena(ValueIdArena &&) = delete;
  ValueIdArena &operator=(ValueIdArena &&) = delete;

private:
  [[nodiscard]] std::optional<ValueRoutes>
  append(std::span<const std::uint32_t> inputs,
         std::span<const std::uint32_t> outputs) {
    constexpr std::size_t IndexLimit =
        std::numeric_limits<std::uint32_t>::max();
    constexpr std::size_t ByteLimit =
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t);
    constexpr std::size_t limit =
        IndexLimit < ByteLimit ? IndexLimit : ByteLimit;
    if (outputs.empty() || inputs.size() > limit || outputs.size() > limit ||
        aliases(inputs) || aliases(outputs) || size_ > limit - inputs.size() ||
        size_ + inputs.size() > limit - outputs.size()) {
      return std::nullopt;
    }
    const std::size_t input_offset = size_;
    const std::size_t output_offset = input_offset + inputs.size();
    const std::size_t required = output_offset + outputs.size();
    if (required > capacity_) {
      std::size_t capacity = capacity_;
      if (capacity == 0u) {
        capacity = required;
      } else {
        const std::size_t half = capacity / 2u + capacity % 2u;
        capacity = capacity > limit - half ? limit : capacity + half;
        if (capacity < required) {
          capacity = required;
        }
      }
      std::unique_ptr<std::uint32_t[]> storage{new std::uint32_t[capacity]};
      if (size_ != 0u) {
        std::copy_n(ids_.get(), size_, storage.get());
      }
      std::copy(inputs.begin(), inputs.end(), storage.get() + input_offset);
      std::copy(outputs.begin(), outputs.end(), storage.get() + output_offset);
      ids_ = std::move(storage);
      capacity_ = capacity;
    } else {
      std::copy(inputs.begin(), inputs.end(), ids_.get() + input_offset);
      std::copy(outputs.begin(), outputs.end(), ids_.get() + output_offset);
    }
    size_ = required;
    return ValueRoutes{
        .inputs = ValueIdRange{static_cast<std::uint32_t>(input_offset),
                               static_cast<std::uint32_t>(inputs.size())},
        .outputs = ValueIdRange{static_cast<std::uint32_t>(output_offset),
                                static_cast<std::uint32_t>(outputs.size())},
    };
  }

public:
  // Route IDs become visible only inside the callback that publishes their
  // owning step. The transaction cannot escape this frame: callback failure
  // retains any grown allocation but restores the exact logical prefix. One
  // callback-scoped publication is the sole mutation authority for this arena.
  template <class Publish>
  [[nodiscard]] bool publish(const std::span<const std::uint32_t> inputs,
                             const std::span<const std::uint32_t> outputs,
                             Publish &&publish_step) {
    if (transaction_active_) {
      return false;
    }
    const std::size_t mark = size_;
    const std::optional<ValueRoutes> routes = append(inputs, outputs);
    if (!routes) {
      return false;
    }
    transaction_active_ = true;
    try {
      std::invoke(std::forward<Publish>(publish_step), *routes);
    } catch (...) {
      rollback(mark);
      throw;
    }
    transaction_active_ = false;
    return true;
  }

  [[nodiscard]] std::span<const std::uint32_t>
  view(const ValueIdRange range) const noexcept {
    if (!valid(range)) {
      return {};
    }
    return range.count == 0u ? std::span<const std::uint32_t>{}
                             : std::span<const std::uint32_t>{
                                   ids_.get() + range.offset, range.count};
  }

  [[nodiscard]] bool valid(const ValueIdRange range) const noexcept {
    return range.offset <= size_ && range.count <= size_ - range.offset;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
  void rollback(const std::size_t mark) noexcept {
    size_ = mark;
    transaction_active_ = false;
  }

  [[nodiscard]] bool
  aliases(const std::span<const std::uint32_t> values) const noexcept {
    if (values.empty() || size_ == 0u) {
      return false;
    }
    const std::less<const std::uint32_t *> less;
    const std::uint32_t *const begin = ids_.get();
    const std::uint32_t *const end = begin + size_;
    const std::uint32_t *const values_begin = values.data();
    const std::uint32_t *const values_end = values_begin + values.size();
    return less(values_begin, end) && less(begin, values_end);
  }

  std::unique_ptr<std::uint32_t[]> ids_;
  std::size_t size_{};
  std::size_t capacity_{};
  bool transaction_active_{};
};

} // namespace rund::compute::detail
