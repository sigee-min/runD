#pragma once

#include "registry/cache_model.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace rund::compute::detail::residency {

struct PersistRequest final {
  CacheKey key{};
  std::uint64_t backing_offset{};
  std::size_t bytes{};
  std::size_t frame_offset{};
  const std::byte *frame{};
  std::uint32_t physical_frame{};
};

struct PersistedPage final {
  CacheKey key{};
  std::uint64_t backing_offset{};
  std::size_t bytes{};
  std::uint32_t physical_frame{};
};

struct PersistReceipt final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  std::span<const PersistedPage> pages;
  std::uint64_t token{};
  std::uint64_t io_ns{};
  bool may_write{};
};

// One fixed metadata worker for Host-output -> backing persistence. Payload is
// retained by the Authority-owned Host frame; this worker never owns or
// allocates page bytes after configure().
class Persister final {
public:
  Persister() = default;
  ~Persister();
  Persister(const Persister &) = delete;
  Persister &operator=(const Persister &) = delete;

  [[nodiscard]] bool configure(std::uint64_t frame_bytes,
                               std::uint32_t capacity) noexcept;
  [[nodiscard]] bool submit(VirtualBacking &,
                            std::span<const PersistRequest>,
                            std::uint64_t token) noexcept;
  [[nodiscard]] PersistReceipt wait() noexcept;
  [[nodiscard]] bool quiescent() const noexcept;
  [[nodiscard]] std::uint64_t retained_host_bytes() const noexcept;

private:
  enum class State : std::uint8_t { Empty, Idle, Pending, Ready, Stop };

  void work() noexcept;

  mutable std::mutex gate_;
  std::condition_variable ready_;
  std::condition_variable pending_;
  std::thread worker_;
  std::vector<PersistRequest> requests_;
  std::vector<PersistedPage> pages_;
  VirtualBacking *backing_{};
  std::uint64_t frame_bytes_{};
  std::size_t count_{};
  std::uint64_t token_{};
  Status status_{Status::success()};
  std::uint64_t io_ns_{};
  bool may_write_{};
  State state_{State::Empty};
};

struct GraphPersistRing final {
  std::array<Persister, 2u> slots;
};

} // namespace rund::compute::detail::residency
