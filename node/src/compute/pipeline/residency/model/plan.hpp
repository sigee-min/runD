#pragma once

#include "graph.hpp"

namespace rund::compute::detail::residency {

class ResidencyPlan;
struct PlanResult;

class ResidencyPlan final {
public:
  ResidencyPlan() = default;
  ResidencyPlan(const ResidencyPlan &) = delete;
  ResidencyPlan &operator=(const ResidencyPlan &) = delete;
  ResidencyPlan(ResidencyPlan &&) noexcept = default;
  ResidencyPlan &operator=(ResidencyPlan &&) noexcept = default;

  [[nodiscard]] std::uint64_t page_bytes() const noexcept {
    return page_bytes_;
  }
  enum class Kind : std::uint8_t { None, Stream, TiledGraph };
  [[nodiscard]] bool streamed() const noexcept { return kind_ == Kind::Stream; }
  [[nodiscard]] bool graph_tiled() const noexcept {
    return kind_ == Kind::TiledGraph;
  }
  [[nodiscard]] const StreamPlan &stream() const noexcept { return stream_; }
  [[nodiscard]] const TiledGraphPlan &tiled_graph() const noexcept {
    return tiled_graph_;
  }
  [[nodiscard]] std::uint32_t frame_capacity() const noexcept {
    return frame_capacity_;
  }
  [[nodiscard]] std::uint64_t prefetch_distance() const noexcept {
    return streamed()      ? stream_.prefetch_distance()
           : graph_tiled() ? tiled_graph_.prefetch_distance()
                           : 0u;
  }
  [[nodiscard]] Identity identity() const noexcept { return identity_; }

private:
  friend PlanResult PlanResidency(const StreamPlanInput &input) noexcept;
  friend PlanResult PlanResidency(const TiledGraphPlanInput &input) noexcept;
  ResidencyPlan(std::uint64_t page_bytes, StreamPlan stream,
                Identity identity) noexcept;
  ResidencyPlan(std::uint64_t page_bytes, TiledGraphPlan tiled_graph,
                Identity identity) noexcept;

  Kind kind_{Kind::None};
  std::uint64_t page_bytes_{};
  std::uint32_t frame_capacity_{};
  StreamPlan stream_{};
  TiledGraphPlan tiled_graph_{};
  Identity identity_{};
};

struct PlanResult final {
  Failure failure{Failure::Invalid};
  ResidencyPlan plan{};
  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == Failure::None;
  }
};

} // namespace rund::compute::detail::residency
