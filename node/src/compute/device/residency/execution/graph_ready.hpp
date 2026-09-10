#pragma once

#include "graph_forecast.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::residency::execution {

// Authority-authenticated HostReady receipt after the backing callback has
// returned and its Forecast lease has been retired. The resident Host rows
// remain protected by their planner-sealed pin through the consumer epoch;
// this fixed descriptor therefore needs no live epoch slot of its own.
class GraphReady final {
public:
  GraphReady() = default;
  GraphReady(const GraphReady &) = delete;
  GraphReady &operator=(const GraphReady &) = delete;
  GraphReady(GraphReady &&other) noexcept { move_from(other); }
  GraphReady &operator=(GraphReady &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr && plan_owner_ != nullptr && page_count_ != 0u;
  }
  [[nodiscard]] Identity plan() const noexcept { return plan_; }
  [[nodiscard]] std::uint64_t coordinate() const noexcept {
    return coordinate_;
  }
  [[nodiscard]] std::span<const GraphForecastPage> pages() const noexcept {
    return {pages_.data(), page_count_};
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::GraphForecastOwner;
  friend class ::rund::compute::detail::residency::GraphPromoteOwner;

  void move_from(GraphReady &) noexcept;
  void clear() noexcept;

  const Authority *owner_{};
  std::shared_ptr<const ResidencyPlan> plan_owner_{};
  std::array<GraphForecastPage, GraphForecastCapacity> pages_{};
  Identity plan_{};
  std::uint64_t coordinate_{};
  std::size_t page_count_{};
};

} // namespace rund::compute::detail::residency::execution
