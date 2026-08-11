#pragma once

#include "../pipeline/residency/model.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

namespace rund::compute::detail::residency {

struct CacheKey final {
  std::uint64_t backing{};
  std::uint64_t version{};
  // Zero identifies the complete backing. Nonzero distinguishes a boundary
  // materialization whose clamped values depend on an active prefix extent.
  std::uint64_t extent{};
  std::uint64_t materialization_hi{};
  std::uint64_t materialization_lo{};
  std::uint64_t page{};

  [[nodiscard]] constexpr bool
  operator==(const CacheKey &) const noexcept = default;
  [[nodiscard]] constexpr bool operator<(const CacheKey &other) const noexcept {
    if (backing != other.backing) {
      return backing < other.backing;
    }
    if (version != other.version) {
      return version < other.version;
    }
    if (extent != other.extent) {
      return extent < other.extent;
    }
    if (materialization_hi != other.materialization_hi) {
      return materialization_hi < other.materialization_hi;
    }
    if (materialization_lo != other.materialization_lo) {
      return materialization_lo < other.materialization_lo;
    }
    return page < other.page;
  }
};

struct CacheUse final {
  CacheKey key{};
  Access access{Access::Read};
  std::uint64_t next_use{};
};

enum class FrameState : std::uint8_t {
  Empty,
  HostReady,
  Mapping,
  DeviceReady,
  Pinned,
  Dirty,
  Writeback,
};

struct CacheBinding final {
  CacheKey key{};
  std::uint32_t frame{};
  Access access{Access::Read};
  bool fetch{};
};

struct CacheTransition final {
  CacheKey key{};
  std::uint32_t frame{};
  TransitionKind kind{TransitionKind::Fetch};
};

struct EpochLease final {
  std::span<const CacheBinding> bindings;
  std::span<const CacheTransition> transitions;
  std::uint64_t token{};
};

enum class AuthorityFailure : std::uint8_t {
  None,
  Invalid,
  Busy,
  Capacity,
};

struct AuthorityResult final {
  AuthorityFailure failure{AuthorityFailure::Invalid};
  EpochLease lease{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AuthorityFailure::None;
  }
};

class Authority final {
public:
  struct Frame final {
    CacheKey key{};
    std::uint64_t next_use{};
    FrameState state{FrameState::Empty};
    bool dirty{};
  };

  Authority() = default;
  Authority(const Authority &) = delete;
  Authority &operator=(const Authority &) = delete;

  [[nodiscard]] bool configure(std::uint64_t page_bytes,
                               std::uint32_t frame_capacity) noexcept;
  [[nodiscard]] AuthorityResult begin(std::span<const CacheUse> uses) noexcept;
  // Freezes every dirty frame into one terminal writeback lease. No new epoch
  // may begin until complete() publishes or rolls back this drain.
  [[nodiscard]] AuthorityResult drain_dirty() noexcept;
  [[nodiscard]] bool complete(std::uint64_t token, bool success) noexcept;
  // A backing write may have changed only an unknown prefix. Discard every
  // dirty frame named by the active lease so no later backing can inherit
  // those bytes after a failed terminal drain.
  [[nodiscard]] bool discard(std::uint64_t token) noexcept;
  [[nodiscard]] bool probe(std::span<const CacheKey> keys,
                           std::span<std::uint8_t> resident) const noexcept;

  [[nodiscard]] std::uint64_t page_bytes() const noexcept;
  [[nodiscard]] std::uint32_t frame_capacity() const noexcept;

private:
  mutable std::mutex gate_;
  std::uint64_t page_bytes_{};
  std::vector<Frame> frames_;
  std::vector<Frame> rollback_;
  std::vector<CacheBinding> bindings_;
  std::vector<CacheTransition> transitions_;
  std::uint64_t next_token_{1u};
  std::uint64_t active_token_{};
  bool active_drain_{};
};

} // namespace rund::compute::detail::residency
