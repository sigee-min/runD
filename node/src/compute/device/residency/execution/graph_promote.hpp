#pragma once

#include "graph_forecast.hpp"
#include "graph_ready.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::residency::execution {

// Public Virtual Graph admission owns at most seven ordered external inputs.
// The planner's ninth row is an internal stage value, not another Forecast
// source. One aggregate promotion retains every exact Forecast source
// generation until the destination stage is activated once.
inline constexpr std::size_t GraphPromoteSourceCapacity = 7u;
static_assert(GraphPromoteSourceCapacity == 7u);
inline constexpr std::size_t GraphPromoteCapacity =
    GraphForecastCapacity * GraphPromoteSourceCapacity;

struct GraphPromotePage final {
  PageKey use{};
  CacheKey key{};
  std::uint64_t bytes{};
  std::uint32_t source_frame{};
  std::uint32_t target_frame{};
};

struct GraphPromoteCompletion final {
  CacheKey key{};
  std::uint64_t bytes{};
  std::uint32_t source_frame{};
  std::uint32_t target_frame{};
};

// Callback-return-gated physical Host-input -> Device-input promotion. The
// capability consumes one successful GraphForecast and binds it to the exact
// still-Prepared Graph stage lease. Only a successful release activates that
// destination lease; callers cannot compose unrelated Host and Device rows.
class GraphPromote final {
public:
  GraphPromote() = default;
  GraphPromote(const GraphPromote &) = delete;
  GraphPromote &operator=(const GraphPromote &) = delete;
  GraphPromote(GraphPromote &&other) noexcept { move_from(other); }
  GraphPromote &operator=(GraphPromote &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr && source_count_ != 0u &&
           destination_token_ != 0u && plan_owner_ != nullptr;
  }
  [[nodiscard]] Identity plan() const noexcept { return plan_; }
  [[nodiscard]] std::uint64_t coordinate() const noexcept {
    return coordinate_;
  }
  [[nodiscard]] std::uint64_t destination_token() const noexcept {
    return destination_token_;
  }
  [[nodiscard]] std::size_t source_count() const noexcept {
    return source_count_;
  }
  [[nodiscard]] std::span<const GraphPromotePage> pages() const noexcept {
    return {pages_.data(), page_count_};
  }

private:
  friend class ::rund::compute::detail::residency::GraphPromoteOwner;

  void move_from(GraphPromote &) noexcept;
  void clear() noexcept;

  const Authority *owner_{};
  std::shared_ptr<const ResidencyPlan> plan_owner_{};
  std::array<GraphPromotePage, GraphPromoteCapacity> pages_{};
  std::array<std::uint64_t, GraphPromoteSourceCapacity> source_tokens_{};
  Identity plan_{};
  Status completion_{Status::fail(Reason::CompletionInvalid)};
  std::uint64_t destination_token_{};
  std::uint64_t coordinate_{};
  std::size_t page_count_{};
  std::size_t source_count_{};
  TerminalKind terminal_{TerminalKind::Known};
  bool completion_may_write_{};
  bool terminalled_{};
};

} // namespace rund::compute::detail::residency::execution
