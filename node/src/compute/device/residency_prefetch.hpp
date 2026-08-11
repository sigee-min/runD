#pragma once

#include "residency.hpp"

#include <rund/compute/virtual.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace rund::compute::detail::residency {

struct PrefetchRequest final {
  CacheKey key{};
  std::uint64_t offset{};
  std::size_t bytes{};
  std::size_t target_offset{};
};

struct PrefetchedPage final {
  CacheKey key{};
  std::size_t bytes{};
  std::size_t storage_offset{};
  std::size_t target_offset{};
};

struct PrefetchReceipt final {
  Status status{Status::success()};
  std::span<const PrefetchedPage> pages;
  const std::byte *storage{};
  std::uint64_t io_ns{};
  std::uint64_t wait_ns{};
};

// One cold-created worker per Device-global Pool. Request, range and payload
// storage are fixed at Pool construction; submit/wait performs no allocation.
class Prefetcher final {
public:
  Prefetcher() = default;
  ~Prefetcher();
  Prefetcher(const Prefetcher &) = delete;
  Prefetcher &operator=(const Prefetcher &) = delete;

  [[nodiscard]] bool configure(std::uint64_t page_bytes,
                               std::uint32_t capacity) noexcept;
  [[nodiscard]] bool submit(VirtualBacking &backing,
                            std::span<const PrefetchRequest> requests) noexcept;
  [[nodiscard]] PrefetchReceipt wait() noexcept;
  [[nodiscard]] std::uint64_t storage_bytes() const noexcept;

private:
  enum class State : std::uint8_t { Empty, Idle, Pending, Ready, Stop };

  void work() noexcept;

  mutable std::mutex gate_;
  std::condition_variable ready_;
  std::condition_variable pending_;
  std::thread worker_;
  std::vector<PrefetchRequest> requests_;
  std::vector<PrefetchedPage> pages_;
  std::vector<VirtualRead> ranges_;
  std::vector<std::byte> storage_;
  VirtualBacking *backing_{};
  std::uint64_t page_bytes_{};
  std::size_t count_{};
  Status status_{Status::success()};
  std::uint64_t io_ns_{};
  State state_{State::Empty};
};

} // namespace rund::compute::detail::residency
