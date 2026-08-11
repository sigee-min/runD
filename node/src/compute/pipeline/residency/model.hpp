#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::compute::detail::residency {

enum class Failure : std::uint8_t {
  None,
  Invalid,
  Capacity,
  Infeasible,
};

struct PageRun final {
  std::uint64_t first_page{};
  std::uint64_t page_count{};
};

struct PageKey final {
  std::uint32_t resource{};
  std::uint64_t page{};

  [[nodiscard]] constexpr bool
  operator==(const PageKey &) const noexcept = default;
  [[nodiscard]] constexpr bool operator<(const PageKey &other) const noexcept {
    return resource < other.resource ||
           (resource == other.resource && page < other.page);
  }
};

enum class Access : std::uint8_t {
  Read,
  Write,
  ReadWrite,
};

[[nodiscard]] constexpr bool reads(const Access access) noexcept {
  return access != Access::Write;
}

[[nodiscard]] constexpr bool writes(const Access access) noexcept {
  return access != Access::Read;
}

struct PageUse final {
  PageKey key{};
  Access access{Access::Read};
};

enum class TransitionKind : std::uint8_t {
  Writeback,
  Unmap,
  Fetch,
  Map,
};

struct Transition final {
  PageKey key{};
  std::uint32_t frame{};
  TransitionKind kind{TransitionKind::Fetch};

  [[nodiscard]] constexpr bool
  operator==(const Transition &) const noexcept = default;
};

struct Epoch final {
  std::uint32_t node{};
  std::uint32_t tile{};
  std::size_t first_use{};
  std::size_t use_count{};
  std::size_t first_transition{};
  std::size_t transition_count{};
};

struct Identity final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return hi != 0u || lo != 0u;
  }
  [[nodiscard]] bool operator==(const Identity &) const noexcept = default;
};

struct StreamPlanInput final {
  std::uint64_t page_bytes{};
  std::uint64_t page_count{};
  std::uint64_t requested_frames{};
  std::uint64_t max_frames{};
};

class StreamPlan final {
public:
  constexpr StreamPlan() noexcept = default;
  constexpr StreamPlan(const std::uint64_t page_count,
                       const std::uint64_t frame_capacity) noexcept
      : page_count_(page_count), frame_capacity_(frame_capacity) {}

  [[nodiscard]] constexpr std::uint64_t page_count() const noexcept {
    return page_count_;
  }
  [[nodiscard]] constexpr std::uint64_t frame_capacity() const noexcept {
    return frame_capacity_;
  }
  [[nodiscard]] constexpr std::uint64_t epoch_count() const noexcept {
    return frame_capacity_ == 0u ? 0u
                                 : page_count_ / frame_capacity_ +
                                       static_cast<std::uint64_t>(
                                           page_count_ % frame_capacity_ != 0u);
  }
  [[nodiscard]] bool epoch(std::uint64_t index, PageRun &run) const noexcept;

private:
  std::uint64_t page_count_{};
  std::uint64_t frame_capacity_{};
};

struct DemandEpoch final {
  std::uint32_t node{};
  std::uint32_t tile{};
  std::vector<PageUse> uses;
};

struct GraphPlanInput final {
  std::uint64_t page_bytes{};
  std::uint32_t frame_capacity{};
  std::vector<DemandEpoch> epochs;
};

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
  [[nodiscard]] bool streamed() const noexcept { return epochs_.empty(); }
  [[nodiscard]] const StreamPlan &stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t frame_capacity() const noexcept {
    return frame_capacity_;
  }
  [[nodiscard]] const std::vector<PageUse> &uses() const noexcept {
    return uses_;
  }
  [[nodiscard]] const std::vector<Transition> &transitions() const noexcept {
    return transitions_;
  }
  [[nodiscard]] const std::vector<Epoch> &epochs() const noexcept {
    return epochs_;
  }
  [[nodiscard]] Identity identity() const noexcept { return identity_; }

private:
  friend PlanResult PlanResidency(const StreamPlanInput &input) noexcept;
  friend PlanResult PlanResidency(const GraphPlanInput &input) noexcept;

  ResidencyPlan(std::uint64_t page_bytes, StreamPlan stream,
                Identity identity) noexcept;
  ResidencyPlan(std::uint64_t page_bytes, std::uint32_t frame_capacity,
                std::vector<PageUse> uses, std::vector<Transition> transitions,
                std::vector<Epoch> epochs, Identity identity) noexcept;

  std::uint64_t page_bytes_{};
  std::uint32_t frame_capacity_{};
  StreamPlan stream_{};
  std::vector<PageUse> uses_;
  std::vector<Transition> transitions_;
  std::vector<Epoch> epochs_;
  Identity identity_{};
};

struct PlanResult final {
  Failure failure{Failure::Invalid};
  ResidencyPlan plan;

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == Failure::None;
  }
};

} // namespace rund::compute::detail::residency
