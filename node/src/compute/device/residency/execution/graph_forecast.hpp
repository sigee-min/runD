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
class GraphForecastOwner;
class GraphPromoteOwner;
class ResidencyPlan;

namespace execution {
inline constexpr std::size_t GraphForecastCapacity = 32u;

struct GraphForecastPage final {
  PageKey use{};
  CacheKey key{};
  std::uint64_t offset{};
  std::uint64_t bytes{};
  std::uint32_t frame{};
  bool fetch{};
};

struct GraphForecastCompletion final {
  CacheKey key{};
  std::uint64_t bytes{};
  std::uint32_t frame{};
};

// Callback-return-gated physical HostReady lease for one exact Graph input
// port subset. Authority embeds four empty values as its bounded quarantine;
// a quarantined value retains the cold plan and exact physical identity.
class GraphForecast final {
public:
  GraphForecast() = default;
  GraphForecast(const GraphForecast &) = delete;
  GraphForecast &operator=(const GraphForecast &) = delete;
  GraphForecast(GraphForecast &&other) noexcept { move_from(other); }
  GraphForecast &operator=(GraphForecast &&other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr && token_ != 0u && generation_ != 0u &&
           plan_owner_ != nullptr;
  }
  [[nodiscard]] Identity plan() const noexcept { return plan_; }
  [[nodiscard]] std::uint64_t token() const noexcept { return token_; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] std::uint64_t coordinate() const noexcept {
    return coordinate_;
  }
  [[nodiscard]] bool requires_backing() const noexcept;
  [[nodiscard]] std::span<const GraphForecastPage> pages() const noexcept {
    return {pages_.data(), page_count_};
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::GraphForecastOwner;
  friend class ::rund::compute::detail::residency::GraphPromoteOwner;

  void move_from(GraphForecast &) noexcept;
  void clear() noexcept;

  const Authority *owner_{};
  std::shared_ptr<const ResidencyPlan> plan_owner_{};
  std::array<GraphForecastPage, GraphForecastCapacity> pages_{};
  Identity plan_{};
  Status completion_{Status::fail(Reason::CompletionInvalid)};
  std::uint64_t token_{};
  std::uint64_t generation_{};
  std::uint64_t coordinate_{};
  std::size_t page_count_{};
  TerminalKind terminal_{TerminalKind::Known};
  bool completion_may_write_{};
  bool terminalled_{};
};

} // namespace execution

} // namespace rund::compute::detail::residency
