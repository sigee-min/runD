#pragma once

#include "model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency::execution {

struct Request final {
  std::uint64_t page_count{};
  std::uint32_t frame_capacity{};
  // Copied from the authoritative StreamPlan. It is part of Plan identity so
  // a backend/controller cannot silently substitute another ready horizon.
  std::uint64_t prefetch_distance{};
  Materialization input{};
  // Optional canonical backing-page identity for a centered Window. `input`
  // remains the halo-expanded materialization identity consumed by the
  // existing Direct path; this distinct owner lets Authority cache and pin
  // canonical pages without aliasing expanded frames.
  GraphMaterialization canonical_input{};
  Materialization output{};
  Publication publication{};
  std::array<FrameRegion, BankCapacity> host_input{};
  std::array<FrameRegion, BankCapacity> device_input{};
  std::array<FrameRegion, BankCapacity> device_output{};
  std::array<FrameRegion, BankCapacity> host_output{};
};

struct SealResult;

class Plan final {
public:
  [[nodiscard]] std::uint64_t page_count() const noexcept {
    return request_.page_count;
  }
  [[nodiscard]] std::uint32_t frame_capacity() const noexcept {
    return request_.frame_capacity;
  }
  [[nodiscard]] std::uint64_t epoch_count() const noexcept {
    return epoch_count_;
  }
  [[nodiscard]] std::uint64_t identity() const noexcept { return identity_; }
  [[nodiscard]] std::uint32_t input_resource() const noexcept {
    return request_.input.cache.resource;
  }
  [[nodiscard]] std::uint32_t output_resource() const noexcept {
    return request_.output.cache.resource;
  }
  [[nodiscard]] bool input_bytes(std::uint64_t page,
                                 std::uint64_t &bytes) const noexcept;
  [[nodiscard]] bool input_source(std::uint64_t page,
                                  FetchSource &) const noexcept;
  [[nodiscard]] bool input_reuse(std::uint64_t page,
                                 FetchReuseSource &) const noexcept;
  [[nodiscard]] bool canonical_input_source(std::uint64_t page,
                                            FetchSource &) const noexcept;
  [[nodiscard]] bool
  window_footprint(std::uint64_t epoch,
                   WindowFootprintProjection &) const noexcept;
  [[nodiscard]] bool has_window_footprint() const noexcept {
    return request_.canonical_input.resource != 0u;
  }
  [[nodiscard]] bool input_live_rows(std::size_t bank,
                                     std::uint64_t &rows) const noexcept;
  // True only while every Direct input page can become a complete frame from
  // its exact backing slice and sealed fill policy. Clamp/Clip halo fill is
  // intentionally not inferred from inactive-tail zero fill.
  [[nodiscard]] bool input_sources_materializable() const noexcept;
  [[nodiscard]] bool external_all_or_none() const noexcept {
    return external_all_or_none_;
  }
  [[nodiscard]] const std::array<FrameRegion, BankCapacity> &
  host_input_regions() const noexcept {
    return request_.host_input;
  }
  [[nodiscard]] const std::array<FrameRegion, BankCapacity> &
  device_input_regions() const noexcept {
    return request_.device_input;
  }
  [[nodiscard]] const std::array<FrameRegion, BankCapacity> &
  device_output_regions() const noexcept {
    return request_.device_output;
  }
  [[nodiscard]] const std::array<FrameRegion, BankCapacity> &
  host_output_regions() const noexcept {
    return request_.host_output;
  }
  [[nodiscard]] const Publication &publication() const noexcept {
    return request_.publication;
  }

  [[nodiscard]] bool project(NodeId, Node &) const noexcept;
  [[nodiscard]] bool forecast(std::uint64_t epoch, Forecast &) const noexcept;
  [[nodiscard]] bool
  predecessors(NodeId, std::span<NodeId, PredecessorCapacity> storage,
               std::size_t &count) const noexcept;

private:
  friend struct SealResult;
  friend SealResult seal(const Request &) noexcept;

  Request request_{};
  std::uint64_t epoch_count_{};
  std::uint64_t identity_{};
  bool external_all_or_none_{};
};

struct SealResult final {
  SealFailure failure{SealFailure::Invalid};
  Plan plan{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == SealFailure::None;
  }
};

// Seals one exact invocation into an O(1) recurrent execution plan. No epoch
// node/edge array is retained, so storage is independent of logical page count.
[[nodiscard]] SealResult seal(const Request &) noexcept;

} // namespace rund::compute::detail::residency::execution
