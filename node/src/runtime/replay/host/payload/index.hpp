#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rund::node::replay_detail::payload {

// A fixed, prepared hash index. Linear probing preserves insertion order for
// equal hashes, while the avalanche step keeps the power-of-two table from
// inheriting structure from application hashes. Clear advances an epoch and
// therefore performs O(1) work on every practical scope boundary.
class Index final {
public:
  Index() = default;

  explicit Index(const std::size_t capacity) : capacity_{capacity} {
    if (capacity == 0u) {
      return;
    }
    if (capacity > std::numeric_limits<std::size_t>::max() / 2u) {
      throw std::length_error{"replay_payload_index_capacity_exceeded"};
    }
    const std::size_t required = capacity * 2u;
    std::size_t slots = 1u;
    while (slots < required) {
      if (slots > std::numeric_limits<std::size_t>::max() / 2u) {
        throw std::length_error{"replay_payload_index_capacity_exceeded"};
      }
      slots *= 2u;
    }
    slots_.resize(slots);
    mask_ = slots - 1u;
    epoch_ = 1u;
  }

  Index(const Index &) = delete;
  Index &operator=(const Index &) = delete;
  Index(Index &&) noexcept = default;
  Index &operator=(Index &&) noexcept = default;

  template <class Match>
  [[nodiscard]] std::optional<std::uint32_t>
  find(const std::uint64_t hash, Match &&match) const {
    if (slots_.empty()) {
      return std::nullopt;
    }
    std::size_t slot = avalanche(hash) & mask_;
    for (std::size_t probe = 0u; probe < slots_.size(); ++probe) {
      const Slot &entry = slots_[slot];
      if (entry.epoch != epoch_) {
        return std::nullopt;
      }
      if (entry.hash == hash && match(entry.index)) {
        return entry.index;
      }
      slot = (slot + 1u) & mask_;
    }
    return std::nullopt;
  }

  [[nodiscard]] bool insert(const std::uint64_t hash,
                            const std::uint32_t index) noexcept {
    if (size_ >= capacity_ || slots_.empty()) {
      return false;
    }
    std::size_t slot = avalanche(hash) & mask_;
    for (std::size_t probe = 0u; probe < slots_.size(); ++probe) {
      Slot &entry = slots_[slot];
      if (entry.epoch != epoch_) {
        entry = Slot{.hash = hash, .index = index, .epoch = epoch_};
        ++size_;
        return true;
      }
      slot = (slot + 1u) & mask_;
    }
    return false;
  }

  void clear() noexcept {
    size_ = 0u;
    if (slots_.empty()) {
      return;
    }
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) {
      for (Slot &slot : slots_) {
        slot.epoch = 0u;
      }
      epoch_ = 1u;
      return;
    }
    ++epoch_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  struct Slot final {
    std::uint64_t hash = 0u;
    std::uint32_t index = 0u;
    std::uint64_t epoch = 0u;
  };

  [[nodiscard]] static std::size_t avalanche(std::uint64_t value) noexcept {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return static_cast<std::size_t>(value);
  }

  std::vector<Slot> slots_{};
  std::size_t capacity_ = 0u;
  std::size_t mask_ = 0u;
  std::size_t size_ = 0u;
  std::uint64_t epoch_ = 0u;
};

} // namespace rund::node::replay_detail::payload
