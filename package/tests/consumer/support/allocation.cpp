#include "allocation.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

std::atomic<bool> active{false};
std::atomic<std::uint64_t> allocations{0u};

[[nodiscard]] void *allocate(const std::size_t size) {
  if (active.load(std::memory_order_relaxed)) {
    allocations.fetch_add(1u, std::memory_order_relaxed);
  }
  void *const value = std::malloc(size == 0u ? 1u : size);
  if (value == nullptr) throw std::bad_alloc{};
  return value;
}

[[nodiscard]] void *allocate_aligned(const std::size_t size,
                                     const std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t)) return allocate(size);
  if (active.load(std::memory_order_relaxed)) {
    allocations.fetch_add(1u, std::memory_order_relaxed);
  }
  void *value = nullptr;
  if (posix_memalign(&value, alignment, size == 0u ? alignment : size) != 0 ||
      value == nullptr) {
    throw std::bad_alloc{};
  }
  return value;
}

} // namespace

namespace package_consumer::allocation {

void start() noexcept {
  allocations.store(0u, std::memory_order_relaxed);
  active.store(true, std::memory_order_relaxed);
}

std::uint64_t stop() noexcept {
  active.store(false, std::memory_order_relaxed);
  return allocations.load(std::memory_order_relaxed);
}

} // namespace package_consumer::allocation

void *operator new(const std::size_t size) { return allocate(size); }
void *operator new[](const std::size_t size) { return allocate(size); }
void *operator new(const std::size_t size, const std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void *operator new[](const std::size_t size,
                     const std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<std::size_t>(alignment));
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
void operator delete[](void *value, std::size_t,
                       std::align_val_t) noexcept {
  std::free(value);
}
