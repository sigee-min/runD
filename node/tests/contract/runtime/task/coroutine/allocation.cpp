#include "allocation.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace runtime_task_allocation {
namespace {

std::atomic<bool> active{false};
std::atomic<bool> fail_next{false};
std::atomic<std::uint64_t> count{0u};

void* Allocate(const std::size_t size) {
  if (fail_next.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (active.load(std::memory_order_relaxed)) {
    count.fetch_add(1u, std::memory_order_relaxed);
  }
  void* const value = std::malloc(size == 0u ? 1u : size);
  if (value == nullptr) {
    throw std::bad_alloc();
  }
  return value;
}

void* AllocateAligned(const std::size_t size, const std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t)) {
    return Allocate(size);
  }
  if (fail_next.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (active.load(std::memory_order_relaxed)) {
    count.fetch_add(1u, std::memory_order_relaxed);
  }
  void* value = nullptr;
  if (posix_memalign(&value, alignment,
                     size == 0u ? alignment : size) != 0 ||
      value == nullptr) {
    throw std::bad_alloc();
  }
  return value;
}

} // namespace

void Start() noexcept {
  count.store(0u, std::memory_order_relaxed);
  active.store(true, std::memory_order_release);
}

void Stop() noexcept { active.store(false, std::memory_order_release); }

void FailNext() noexcept { fail_next.store(true, std::memory_order_release); }

std::uint64_t Count() noexcept {
  return count.load(std::memory_order_acquire);
}

} // namespace runtime_task_allocation

void* operator new(const std::size_t size) {
  return runtime_task_allocation::Allocate(size);
}

void* operator new[](const std::size_t size) {
  return runtime_task_allocation::Allocate(size);
}

void* operator new(const std::size_t size, const std::align_val_t alignment) {
  return runtime_task_allocation::AllocateAligned(
      size, static_cast<std::size_t>(alignment));
}

void* operator new[](const std::size_t size,
                     const std::align_val_t alignment) {
  return runtime_task_allocation::AllocateAligned(
      size, static_cast<std::size_t>(alignment));
}

void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }
void operator delete(void* value, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete[](void* value, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete(void* value, std::size_t, std::align_val_t) noexcept {
  std::free(value);
}
void operator delete[](void* value, std::size_t,
                       std::align_val_t) noexcept {
  std::free(value);
}
