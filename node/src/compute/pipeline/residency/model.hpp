#pragma once

#include <cstdint>

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

struct Identity final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return hi != 0u || lo != 0u;
  }
  [[nodiscard]] bool operator==(const Identity &) const noexcept = default;
};

struct PlanInput final {
  std::uint64_t page_bytes{};
  std::uint64_t page_count{};
  std::uint64_t requested_slots{};
  std::uint64_t max_slots{};
};

class LinearPlan final {
public:
  constexpr LinearPlan() noexcept = default;
  constexpr LinearPlan(const std::uint64_t page_count,
                       const std::uint64_t slot_capacity) noexcept
      : page_count_(page_count), slot_capacity_(slot_capacity) {}

  [[nodiscard]] constexpr std::uint64_t page_count() const noexcept {
    return page_count_;
  }
  [[nodiscard]] constexpr std::uint64_t slot_capacity() const noexcept {
    return slot_capacity_;
  }
  [[nodiscard]] constexpr std::uint64_t wave_count() const noexcept {
    return slot_capacity_ == 0u ? 0u
                                : page_count_ / slot_capacity_ +
                                      static_cast<std::uint64_t>(
                                          page_count_ % slot_capacity_ != 0u);
  }
  [[nodiscard]] bool wave(std::uint64_t index, PageRun &run) const noexcept;

private:
  std::uint64_t page_count_{};
  std::uint64_t slot_capacity_{};
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
  [[nodiscard]] const LinearPlan &linear() const noexcept { return linear_; }
  [[nodiscard]] Identity identity() const noexcept { return identity_; }

private:
  friend PlanResult PlanResidency(const PlanInput &input) noexcept;

  ResidencyPlan(std::uint64_t page_bytes, LinearPlan linear,
                Identity identity) noexcept;

  std::uint64_t page_bytes_{};
  LinearPlan linear_{};
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
