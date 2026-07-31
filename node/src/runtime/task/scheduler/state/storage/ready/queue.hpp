#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class ReadyQueue final {
public:
  void configure(const std::size_t capacity) {
    ids_.assign(capacity, 0u);
    next_.assign(capacity, 0u);
    clear();
  }

  void clear() noexcept {
    head_ = 0u;
    tail_ = 0u;
    free_ = ids_.empty() ? 0u : 1u;
    size_ = 0u;
    for (std::size_t index = 0u; index < ids_.size(); ++index) {
      ids_[index] = 0u;
      next_[index] = index + 1u < ids_.size()
                         ? static_cast<std::uint32_t>(index + 2u)
                         : 0u;
    }
  }

  [[nodiscard]] bool empty() const noexcept { return size_ == 0u; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept {
    return ids_.size();
  }

  [[nodiscard]] bool front(std::uint64_t &id) const noexcept {
    if (head_ == 0u) {
      return false;
    }
    id = ids_[head_ - 1u];
    return true;
  }

  [[nodiscard]] bool push_back(const std::uint64_t id) noexcept {
    const std::uint32_t slot = claim(id);
    if (slot == 0u) {
      return false;
    }
    if (tail_ == 0u) {
      head_ = slot;
    } else {
      next_[tail_ - 1u] = slot;
    }
    tail_ = slot;
    return true;
  }

  [[nodiscard]] bool push_front(const std::uint64_t id) noexcept {
    const std::uint32_t slot = claim(id);
    if (slot == 0u) {
      return false;
    }
    next_[slot - 1u] = head_;
    head_ = slot;
    if (tail_ == 0u) {
      tail_ = slot;
    }
    return true;
  }

  [[nodiscard]] bool pop_front(std::uint64_t& id) noexcept {
    if (head_ == 0u) {
      return false;
    }
    const std::uint32_t slot = head_;
    id = ids_[slot - 1u];
    head_ = next_[slot - 1u];
    if (head_ == 0u) tail_ = 0u;
    release(slot);
    return true;
  }

  template <class Predicate>
  [[nodiscard]] bool take_first(Predicate&& predicate,
                                std::uint64_t& id) noexcept {
    std::uint32_t previous = 0u;
    std::uint32_t slot = head_;
    while (slot != 0u) {
      const std::uint64_t candidate = ids_[slot - 1u];
      if (predicate(candidate)) {
        id = candidate;
        unlink(previous, slot);
        return true;
      }
      previous = slot;
      slot = next_[slot - 1u];
    }
    return false;
  }

  [[nodiscard]] bool remove(const std::uint64_t id) noexcept {
    std::uint64_t removed = 0u;
    return take_first([id](const std::uint64_t candidate) {
      return candidate == id;
    }, removed);
  }

private:
  [[nodiscard]] std::uint32_t claim(const std::uint64_t id) noexcept {
    if (id == 0u || free_ == 0u) return 0u;
    const std::uint32_t slot = free_;
    free_ = next_[slot - 1u];
    ids_[slot - 1u] = id;
    next_[slot - 1u] = 0u;
    ++size_;
    return slot;
  }

  void release(const std::uint32_t slot) noexcept {
    ids_[slot - 1u] = 0u;
    next_[slot - 1u] = free_;
    free_ = slot;
    --size_;
  }

  void unlink(const std::uint32_t previous,
              const std::uint32_t slot) noexcept {
    const std::uint32_t following = next_[slot - 1u];
    if (previous == 0u) {
      head_ = following;
    } else {
      next_[previous - 1u] = following;
    }
    if (tail_ == slot) tail_ = previous;
    release(slot);
  }

  std::vector<std::uint64_t> ids_{};
  std::vector<std::uint32_t> next_{};
  std::uint32_t head_ = 0u;
  std::uint32_t tail_ = 0u;
  std::uint32_t free_ = 0u;
  std::size_t size_ = 0u;
};
