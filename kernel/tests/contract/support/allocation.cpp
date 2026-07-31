#include "contract/support/allocation.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace kernel_contract_test::memory_allocation {
namespace {

std::atomic<bool> enabled{false};
std::atomic<bool> fail_next{false};
std::atomic<std::uint64_t> allocations{0u};

} // namespace

void *Allocate(const std::size_t size) {
  if (fail_next.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (enabled.load(std::memory_order_relaxed)) {
    allocations.fetch_add(1u, std::memory_order_relaxed);
  }
  void *const out = std::malloc(size == 0u ? 1u : size);
  if (out == nullptr) {
    throw std::bad_alloc();
  }
  return out;
}

void *AllocateAligned(const std::size_t size, const std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t)) {
    return Allocate(size);
  }
  if (fail_next.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (enabled.load(std::memory_order_relaxed)) {
    allocations.fetch_add(1u, std::memory_order_relaxed);
  }
  void *out = nullptr;
  if (posix_memalign(&out, alignment, size == 0u ? alignment : size) != 0 ||
      out == nullptr) {
    throw std::bad_alloc();
  }
  return out;
}

void Reset() noexcept {
  allocations.store(0u, std::memory_order_relaxed);
  fail_next.store(false, std::memory_order_relaxed);
  enabled.store(true, std::memory_order_relaxed);
}

void Stop() noexcept {
  enabled.store(false, std::memory_order_relaxed);
  fail_next.store(false, std::memory_order_relaxed);
}

void FailNext() noexcept { fail_next.store(true, std::memory_order_relaxed); }

std::uint64_t Count() noexcept {
  return allocations.load(std::memory_order_relaxed);
}

} // namespace kernel_contract_test::memory_allocation

void *operator new(std::size_t size) {
  return kernel_contract_test::memory_allocation::Allocate(size);
}

void *operator new[](std::size_t size) {
  return kernel_contract_test::memory_allocation::Allocate(size);
}

void *operator new(std::size_t size, std::align_val_t alignment) {
  return kernel_contract_test::memory_allocation::AllocateAligned(
      size, static_cast<std::size_t>(alignment));
}

void *operator new[](std::size_t size, std::align_val_t alignment) {
  return kernel_contract_test::memory_allocation::AllocateAligned(
      size, static_cast<std::size_t>(alignment));
}

void operator delete(void *ptr) noexcept { std::free(ptr); }

void operator delete[](void *ptr) noexcept { std::free(ptr); }

void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }

void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }

void operator delete(void *ptr, std::align_val_t) noexcept { std::free(ptr); }

void operator delete[](void *ptr, std::align_val_t) noexcept { std::free(ptr); }

void operator delete(void *ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void *ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}
