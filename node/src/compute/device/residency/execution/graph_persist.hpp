#pragma once

#include "../../../pipeline/residency/model.hpp"
#include "../registry/cache_model.hpp"
#include "../registry/credentials/cpu.hpp"
#include "evidence.hpp"
#include "graph_persist/capacity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund::compute::detail::residency {
class GraphPersistOwner;
}

namespace rund::compute::detail::residency::execution {

struct GraphPersistPage final {
  PageKey use{};
  CacheKey key{};
  std::uint64_t backing_offset{};
  std::uint64_t frame_offset{};
  std::uint64_t bytes{};
  std::uint32_t frame{};
};

struct GraphPersistCompletion final {
  CacheKey key{};
  std::uint64_t backing_offset{};
  std::uint64_t bytes{};
  std::uint32_t frame{};
};

using GraphPersistIdentity =
    ::rund::compute::detail::residency::GraphPersistIdentity;

// Callback-return-gated physical Host-output -> backing persistence. The
// capability owns one of two fixed Authority slots, so a following Graph Drain
// may fill the other Host bank while an earlier backing callback remains live.
class GraphPersist final {
public:
  GraphPersist() = default;
  GraphPersist(const GraphPersist &) = delete;
  GraphPersist &operator=(const GraphPersist &) = delete;
  GraphPersist(GraphPersist &&other) noexcept { move_from(other); }
  GraphPersist &operator=(GraphPersist &&other) noexcept = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr && token_ != 0u && generation_ != 0u &&
           coordinate_ != 0u && plan_owner_ != nullptr && region_.count != 0u;
  }
  [[nodiscard]] Identity plan() const noexcept { return plan_; }
  [[nodiscard]] std::uint64_t coordinate() const noexcept {
    return coordinate_;
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] std::uint64_t token() const noexcept { return token_; }
  [[nodiscard]] std::uint64_t book_domain() const noexcept {
    return book_domain_;
  }
  [[nodiscard]] FrameRegion region() const noexcept { return region_; }
  [[nodiscard]] const GraphPersistIdentity &identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] bool unknown() const noexcept {
    return terminal_ == TerminalKind::UnknownMayWrite;
  }
  // The CPU receipt book seals its domain before Authority issue.  This is a
  // one-way identity bind; issue never accepts a zero or replacement domain.
  [[nodiscard]] bool bind_book_domain(const std::uint64_t domain) noexcept {
    if (owner_ != nullptr || book_domain_ != 0u || domain == 0u ||
        domain == std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
    book_domain_ = domain;
    return true;
  }
  [[nodiscard]] bool
  bind_identity(const GraphPersistIdentity &identity) noexcept {
    if (owner_ != nullptr || identity_ != GraphPersistIdentity{} ||
        !identity.valid()) {
      return false;
    }
    identity_ = identity;
    return true;
  }
  [[nodiscard]] std::span<const GraphPersistPage> pages() const noexcept {
    return {pages_.data(), page_count_};
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::GraphPersistOwner;
  friend class ::rund::compute::detail::graph_reduce::CpuReceiptBook;

  void move_from(GraphPersist &) noexcept;
  void clear() noexcept;

  const Authority *owner_{};
  std::shared_ptr<const ResidencyPlan> plan_owner_{};
  std::array<GraphPersistPage, GraphPersistCapacity> pages_{};
  Identity plan_{};
  GraphPersistIdentity identity_{};
  Status completion_{Status::fail(Reason::CompletionInvalid)};
  std::uint64_t token_{};
  std::uint64_t generation_{};
  std::uint64_t coordinate_{};
  std::uint64_t book_domain_{};
  FrameRegion region_{};
  std::size_t page_count_{};
  TerminalKind terminal_{TerminalKind::Known};
  bool completion_may_write_{};
  bool identity_bad_{};
  bool terminalled_{};
};

} // namespace rund::compute::detail::residency::execution
