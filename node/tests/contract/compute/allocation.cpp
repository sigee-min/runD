#include "allocation.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>

namespace node_compute_allocation {
namespace {

std::atomic<bool> active{false};
std::atomic<std::uint64_t> count{0u};
std::atomic<std::uint64_t> bytes{0u};
thread_local bool fail_next = false;
constexpr std::uint64_t no_scheduled_failure =
    std::numeric_limits<std::uint64_t>::max();
thread_local std::uint64_t successful_allocations_before_failure =
    no_scheduled_failure;

void CheckFailure() {
  if (fail_next) {
    fail_next = false;
    successful_allocations_before_failure = no_scheduled_failure;
    throw std::bad_alloc();
  }
  if (successful_allocations_before_failure == 0u) {
    successful_allocations_before_failure = no_scheduled_failure;
    throw std::bad_alloc();
  }
  if (successful_allocations_before_failure != no_scheduled_failure) {
    --successful_allocations_before_failure;
  }
}

void *Allocate(const std::size_t size) {
  CheckFailure();
  if (active.load(std::memory_order_relaxed)) {
    count.fetch_add(1u, std::memory_order_relaxed);
    bytes.fetch_add(size == 0u ? 1u : size, std::memory_order_relaxed);
  }
  void *const value = std::malloc(size == 0u ? 1u : size);
  if (value == nullptr) {
    throw std::bad_alloc();
  }
  return value;
}

void *AllocateAligned(const std::size_t size, const std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t)) {
    return Allocate(size);
  }
  CheckFailure();
  if (active.load(std::memory_order_relaxed)) {
    count.fetch_add(1u, std::memory_order_relaxed);
    bytes.fetch_add(size == 0u ? alignment : size, std::memory_order_relaxed);
  }
  void *value = nullptr;
  if (posix_memalign(&value, alignment, size == 0u ? alignment : size) != 0 ||
      value == nullptr) {
    throw std::bad_alloc();
  }
  return value;
}

} // namespace

void Start() noexcept {
  count.store(0u, std::memory_order_relaxed);
  bytes.store(0u, std::memory_order_relaxed);
  active.store(true, std::memory_order_relaxed);
}

void Stop() noexcept { active.store(false, std::memory_order_relaxed); }

void FailNext() noexcept {
  fail_next = true;
  successful_allocations_before_failure = no_scheduled_failure;
}

void FailAfter(const std::uint64_t successful_allocations) noexcept {
  fail_next = false;
  successful_allocations_before_failure = successful_allocations;
}

void ClearFailure() noexcept {
  fail_next = false;
  successful_allocations_before_failure = no_scheduled_failure;
}

std::uint64_t Count() noexcept { return count.load(std::memory_order_relaxed); }

std::uint64_t Bytes() noexcept { return bytes.load(std::memory_order_relaxed); }

} // namespace node_compute_allocation

void *operator new(const std::size_t size) {
  return node_compute_allocation::Allocate(size);
}

void *operator new[](const std::size_t size) {
  return node_compute_allocation::Allocate(size);
}

void *operator new(const std::size_t size, const std::align_val_t alignment) {
  return node_compute_allocation::AllocateAligned(
      size, static_cast<std::size_t>(alignment));
}

void *operator new[](const std::size_t size, const std::align_val_t alignment) {
  return node_compute_allocation::AllocateAligned(
      size, static_cast<std::size_t>(alignment));
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
