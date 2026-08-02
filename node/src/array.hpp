#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rund::node::detail {

// One array interface with exactly one storage authority. Public/cold paths
// own a vector; sealed preparation paths bind a typed arena slice. Borrowed
// storage is constructed and destroyed by that arena and is never reallocated
// by this view.
template <class T> class PreparedArray final {
public:
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  PreparedArray() noexcept = default;

  PreparedArray(const PreparedArray &other)
      : owned_(other.begin(), other.end()) {}

  PreparedArray &operator=(const PreparedArray &other) {
    if (this != &other) {
      require_owned();
      owned_.assign(other.begin(), other.end());
    }
    return *this;
  }

  PreparedArray(PreparedArray &&other) noexcept
      : owned_(std::move(other.owned_)), borrowed_(other.borrowed_),
        borrowed_size_(other.borrowed_size_),
        borrowed_mode_(other.borrowed_mode_) {
    other.borrowed_ = {};
    other.borrowed_size_ = 0u;
    other.borrowed_mode_ = false;
  }

  PreparedArray &operator=(PreparedArray &&other) {
    if (this != &other) {
      require_owned();
      owned_ = std::move(other.owned_);
      borrowed_ = other.borrowed_;
      borrowed_size_ = other.borrowed_size_;
      borrowed_mode_ = other.borrowed_mode_;
      other.borrowed_ = {};
      other.borrowed_size_ = 0u;
      other.borrowed_mode_ = false;
    }
    return *this;
  }

  PreparedArray &operator=(std::vector<T> values) {
    require_owned();
    owned_ = std::move(values);
    return *this;
  }

  [[nodiscard]] bool bind(const std::span<T> storage,
                          const std::size_t initial_size = 0u) noexcept {
    if (!owned_.empty() || owned_.capacity() != 0u || borrowed_mode_ ||
        initial_size > storage.size()) {
      return false;
    }
    borrowed_ = storage;
    borrowed_size_ = initial_size;
    borrowed_mode_ = true;
    return true;
  }

  [[nodiscard]] bool borrowed() const noexcept { return borrowed_mode_; }

  [[nodiscard]] std::size_t size() const noexcept {
    return borrowed() ? borrowed_size_ : owned_.size();
  }
  [[nodiscard]] std::size_t capacity() const noexcept {
    return borrowed() ? borrowed_.size() : owned_.capacity();
  }
  [[nodiscard]] bool empty() const noexcept { return size() == 0u; }

  [[nodiscard]] T *data() noexcept {
    return borrowed() ? borrowed_.data() : owned_.data();
  }
  [[nodiscard]] const T *data() const noexcept {
    return borrowed() ? borrowed_.data() : owned_.data();
  }
  [[nodiscard]] iterator begin() noexcept { return data(); }
  [[nodiscard]] const_iterator begin() const noexcept { return data(); }
  [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
  [[nodiscard]] iterator end() noexcept {
    return empty() ? data() : data() + size();
  }
  [[nodiscard]] const_iterator end() const noexcept {
    return empty() ? data() : data() + size();
  }
  [[nodiscard]] const_iterator cend() const noexcept { return end(); }

  [[nodiscard]] T &operator[](const std::size_t index) noexcept {
    return data()[index];
  }
  [[nodiscard]] const T &operator[](const std::size_t index) const noexcept {
    return data()[index];
  }
  [[nodiscard]] T &front() noexcept { return (*this)[0u]; }
  [[nodiscard]] const T &front() const noexcept { return (*this)[0u]; }
  [[nodiscard]] T &back() noexcept { return (*this)[size() - 1u]; }
  [[nodiscard]] const T &back() const noexcept { return (*this)[size() - 1u]; }

  operator std::span<T>() noexcept { return {data(), size()}; }
  operator std::span<const T>() const noexcept { return {data(), size()}; }

  void clear() noexcept {
    if (borrowed()) {
      borrowed_size_ = 0u;
    } else {
      owned_.clear();
    }
  }

  void reserve(const std::size_t count) {
    if (borrowed()) {
      if (count > borrowed_.size()) {
        throw std::length_error("prepared array capacity");
      }
      return;
    }
    owned_.reserve(count);
  }

  void resize(const std::size_t count) {
    if (borrowed()) {
      if (count > borrowed_.size()) {
        throw std::length_error("prepared array capacity");
      }
      if (count > borrowed_size_) {
        std::fill(borrowed_.begin() + borrowed_size_, borrowed_.begin() + count,
                  T{});
      }
      borrowed_size_ = count;
      return;
    }
    owned_.resize(count);
  }

  void push_back(const T &value) {
    if (borrowed()) {
      append_borrowed(value);
    } else {
      owned_.push_back(value);
    }
  }

  void push_back(T &&value) {
    if (borrowed()) {
      append_borrowed(std::move(value));
    } else {
      owned_.push_back(std::move(value));
    }
  }

  template <class... Args> T &emplace_back(Args &&...args) {
    if (borrowed()) {
      if (borrowed_size_ == borrowed_.size()) {
        throw std::length_error("prepared array capacity");
      }
      T &slot = borrowed_[borrowed_size_++];
      slot = T(std::forward<Args>(args)...);
      return slot;
    }
    return owned_.emplace_back(std::forward<Args>(args)...);
  }

  template <class Iterator> void assign(Iterator first, Iterator last) {
    if (!borrowed()) {
      owned_.assign(first, last);
      return;
    }
    const auto count = static_cast<std::size_t>(std::distance(first, last));
    if (count > borrowed_.size()) {
      throw std::length_error("prepared array capacity");
    }
    std::copy(first, last, borrowed_.begin());
    borrowed_size_ = count;
  }

  void swap(PreparedArray &other) {
    require_owned();
    other.require_owned();
    owned_.swap(other.owned_);
  }

  [[nodiscard]] std::uint64_t owned_bytes() const noexcept {
    return borrowed()
               ? 0u
               : static_cast<std::uint64_t>(owned_.capacity()) * sizeof(T);
  }

  [[nodiscard]] bool operator==(const PreparedArray &other) const noexcept {
    return std::equal(begin(), end(), other.begin(), other.end());
  }

private:
  void require_owned() const {
    if (borrowed()) {
      throw std::logic_error("borrowed prepared array is immutable in size");
    }
  }

  template <class Value> void append_borrowed(Value &&value) {
    if (borrowed_size_ == borrowed_.size()) {
      throw std::length_error("prepared array capacity");
    }
    borrowed_[borrowed_size_++] = std::forward<Value>(value);
  }

  std::vector<T> owned_{};
  std::span<T> borrowed_{};
  std::size_t borrowed_size_{};
  bool borrowed_mode_{};
};

} // namespace rund::node::detail
