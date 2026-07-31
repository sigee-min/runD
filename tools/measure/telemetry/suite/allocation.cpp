#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace allocation {
namespace {

std::atomic<bool> active{false};
std::atomic<std::uint64_t> count{0u};

[[nodiscard]] void *Take(const std::size_t size) {
  if (active.load(std::memory_order_relaxed)) {
    count.fetch_add(1u, std::memory_order_relaxed);
  }
  void *const value = std::malloc(size == 0u ? 1u : size);
  if (value == nullptr) {
    throw std::bad_alloc{};
  }
  return value;
}

[[nodiscard]] void *Take(const std::size_t size, const std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t)) {
    return Take(size);
  }
  if (active.load(std::memory_order_relaxed)) {
    count.fetch_add(1u, std::memory_order_relaxed);
  }
  void *value = nullptr;
  if (::posix_memalign(&value, alignment, size == 0u ? alignment : size) != 0 ||
      value == nullptr) {
    throw std::bad_alloc{};
  }
  return value;
}

} // namespace

void Start() noexcept {
  count.store(0u, std::memory_order_relaxed);
  active.store(true, std::memory_order_release);
}

[[nodiscard]] std::uint64_t Stop() noexcept {
  active.store(false, std::memory_order_release);
  return count.load(std::memory_order_acquire);
}

} // namespace allocation

void *operator new(const std::size_t size) { return allocation::Take(size); }
void *operator new[](const std::size_t size) { return allocation::Take(size); }
void *operator new(const std::size_t size, const std::align_val_t alignment) {
  return allocation::Take(size, static_cast<std::size_t>(alignment));
}
void *operator new[](const std::size_t size, const std::align_val_t alignment) {
  return allocation::Take(size, static_cast<std::size_t>(alignment));
}
void *operator new(const std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return allocation::Take(size);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](const std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return allocation::Take(size);
  } catch (...) {
    return nullptr;
  }
}
void *operator new(const std::size_t size, const std::align_val_t alignment,
                   const std::nothrow_t &) noexcept {
  try {
    return allocation::Take(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](const std::size_t size, const std::align_val_t alignment,
                     const std::nothrow_t &) noexcept {
  try {
    return allocation::Take(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void *value) noexcept { std::free(value); }
void operator delete[](void *value) noexcept { std::free(value); }
void operator delete(void *value, std::size_t) noexcept { std::free(value); }
void operator delete[](void *value, std::size_t) noexcept { std::free(value); }
void operator delete(void *value, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete[](void *value, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete(void *value, std::size_t, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete[](void *value, std::size_t, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete(void *value, const std::nothrow_t &) noexcept {
  std::free(value);
}
void operator delete[](void *value, const std::nothrow_t &) noexcept {
  std::free(value);
}
void operator delete(void *value, std::align_val_t,
                     const std::nothrow_t &) noexcept {
  std::free(value);
}
void operator delete[](void *value, std::align_val_t,
                       const std::nothrow_t &) noexcept {
  std::free(value);
}
