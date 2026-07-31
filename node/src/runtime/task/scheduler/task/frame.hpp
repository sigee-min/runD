#pragma once

#include <rund/task/status.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node {

struct FrameLimits final {
  std::uint32_t capacity{};
  std::uint32_t bytes{};
  std::uint32_t alignment{64u};
};

struct FrameLease final {
  void *data{};
  void *authority{};
  std::uint32_t slot{};
  std::uint32_t generation{};
  std::uint32_t bytes{};
  std::uint32_t alignment{};
  std::uint8_t tier{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return data != nullptr;
  }
};

struct FrameStats final {
  std::uint64_t live{};
  std::uint64_t high_water{};
  std::uint64_t allocations{};
  std::uint64_t reuses{};
  std::uint64_t failures{};
  std::uint64_t compact_allocations{};
  std::uint64_t wide_allocations{};
  std::uint64_t resident_slots{};
  std::uint64_t resident_bytes{};
};

class FrameArena final {
public:
  FrameArena() noexcept = default;
  ~FrameArena();
  FrameArena(const FrameArena &) = delete;
  FrameArena &operator=(const FrameArena &) = delete;

  [[nodiscard]] task::Status configure(FrameLimits limits) noexcept;
  void begin_epoch() noexcept;
  void reset() noexcept;
  [[nodiscard]] FrameLease acquire(std::size_t bytes,
                                   std::size_t alignment) noexcept;
  void release(FrameLease lease) noexcept;
  static void release_frame(void *frame) noexcept;
  [[nodiscard]] static std::uint32_t frame_bytes(void *frame) noexcept;
  [[nodiscard]] static bool frame_reused(void *frame) noexcept;
  [[nodiscard]] FrameStats stats() const noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] ReasonCode code() const noexcept;

private:
  struct Header;
  struct Store;

  static void drop(Store *store) noexcept;
  static void release_lease(FrameLease lease) noexcept;

  Store *store_{};
  ReasonCode code_{ReasonCode::TaskFrameNotConfigured};
};

} // namespace rund::node

namespace rund::detail::task::frame {

void Bind(::rund::node::FrameArena *arena) noexcept;
void Block() noexcept;
[[nodiscard]] ::rund::node::FrameArena *Active() noexcept;

} // namespace rund::detail::task::frame
