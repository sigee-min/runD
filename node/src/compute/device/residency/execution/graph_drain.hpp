#pragma once

#include "../../../pipeline/residency/model.hpp"
#include "../registry/cache_model.hpp"
#include "evidence.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::residency {
class Authority;
class GraphDrainOwner;
class ResidencyPlan;
} // namespace rund::compute::detail::residency

namespace rund::compute::detail::residency::execution {

inline constexpr std::size_t GraphDrainCapacity = 32u;

struct GraphDrainPage final {
  PageKey use{};
  CacheKey key{};
  std::uint64_t bytes{};
  std::uint32_t source_frame{};
  std::uint32_t target_frame{};
};

struct GraphDrainCompletion final {
  CacheKey key{};
  std::uint64_t bytes{};
  std::uint32_t source_frame{};
  std::uint32_t target_frame{};
};

// Callback-return-gated physical Device-output -> Host-output migration. One
// capability owns both Authority tokens; callers cannot pair an unrelated
// destination epoch with a source drain or release either frame early.
class GraphDrain final {
public:
  GraphDrain() = default;
  GraphDrain(const GraphDrain &) = delete;
  GraphDrain &operator=(const GraphDrain &) = delete;
  GraphDrain(GraphDrain &&other) noexcept { move_from(other); }
  GraphDrain &operator=(GraphDrain &&other) noexcept = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr && source_token_ != 0u &&
           destination_token_ != 0u && plan_owner_ != nullptr;
  }
  [[nodiscard]] Identity plan() const noexcept { return plan_; }
  [[nodiscard]] std::uint64_t coordinate() const noexcept {
    return coordinate_;
  }
  [[nodiscard]] std::span<const GraphDrainPage> pages() const noexcept {
    return {pages_.data(), page_count_};
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::GraphDrainOwner;

  void move_from(GraphDrain &) noexcept;
  void clear() noexcept;

  const Authority *owner_{};
  std::shared_ptr<const ResidencyPlan> plan_owner_{};
  std::array<GraphDrainPage, GraphDrainCapacity> pages_{};
  Identity plan_{};
  Status completion_{Status::fail(Reason::CompletionInvalid)};
  std::uint64_t source_token_{};
  std::uint64_t destination_token_{};
  std::uint64_t coordinate_{};
  std::size_t page_count_{};
  TerminalKind terminal_{TerminalKind::Known};
  bool completion_may_write_{};
  bool terminalled_{};
};

} // namespace rund::compute::detail::residency::execution
