#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>

namespace rund::node::runtime_detail {

// Vector growth is serialized by the Compute host mutex. Availability has one
// atomic bitmap authority so release never needs that global lock. A set bit
// is free; claim clears the lowest free bit to preserve deterministic reuse.
class SlotSet final {
public:
  void configure(const std::size_t capacity) {
    capacity_ = capacity;
    word_count_ = (capacity + kWordBits - 1u) / kWordBits;
    free_ = std::make_unique<std::atomic<std::uint64_t>[]>(word_count_);
    for (std::size_t index = 0u; index < word_count_; ++index) {
      free_[index].store(0u, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] std::optional<std::size_t>
  claim(const std::size_t allocated) noexcept {
    const std::size_t allocated_words =
        (allocated + kWordBits - 1u) / kWordBits;
    for (std::size_t word_index = 0u;
         word_index < allocated_words && word_index < word_count_;
         ++word_index) {
      std::uint64_t candidates =
          free_[word_index].load(std::memory_order_acquire);
      while (candidates != 0u) {
        const std::size_t bit = std::countr_zero(candidates);
        const std::size_t index = word_index * kWordBits + bit;
        const std::uint64_t claimed =
            candidates & ~(std::uint64_t{1u} << bit);
        if (free_[word_index].compare_exchange_weak(
                candidates, claimed, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          return index;
        }
      }
    }
    if (allocated >= capacity_) {
      return std::nullopt;
    }
    return allocated;
  }

  [[nodiscard]] bool release(const std::size_t index) noexcept {
    if (index >= capacity_) {
      return false;
    }
    const std::size_t word_index = index / kWordBits;
    const std::uint64_t bit =
        std::uint64_t{1u} << static_cast<unsigned>(index % kWordBits);
    const std::uint64_t prior =
        free_[word_index].fetch_or(bit, std::memory_order_release);
    return (prior & bit) == 0u;
  }

  [[nodiscard]] bool claimed(const std::size_t index) const noexcept {
    if (index >= capacity_) {
      return false;
    }
    const std::size_t word_index = index / kWordBits;
    const std::uint64_t bit =
        std::uint64_t{1u} << static_cast<unsigned>(index % kWordBits);
    return (free_[word_index].load(std::memory_order_acquire) & bit) == 0u;
  }

private:
  static constexpr std::size_t kWordBits = 64u;

  std::unique_ptr<std::atomic<std::uint64_t>[]> free_{};
  std::size_t word_count_ = 0u;
  std::size_t capacity_ = 0u;
};

} // namespace rund::node::runtime_detail
