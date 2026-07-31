#pragma once

#include "model.hpp"

namespace rund::measure::telemetry {

[[nodiscard]] constexpr std::size_t Index(const Setting value) noexcept {
  return static_cast<std::size_t>(value);
}

[[nodiscard]] constexpr std::size_t Index(const Operation value) noexcept {
  return static_cast<std::size_t>(value);
}

[[nodiscard]] constexpr std::uint64_t Add(const std::uint64_t left,
                                          const std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

[[nodiscard]] constexpr std::uint64_t
Multiply(const std::uint64_t value, const std::uint64_t factor) noexcept {
  return factor != 0u &&
                 value > std::numeric_limits<std::uint64_t>::max() / factor
             ? std::numeric_limits<std::uint64_t>::max()
             : value * factor;
}

[[nodiscard]] constexpr std::uint64_t Count(const std::size_t value) noexcept {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] inline bool CpuTime(std::uint64_t &value) noexcept {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_utime.tv_sec < 0 ||
      usage.ru_utime.tv_usec < 0 || usage.ru_stime.tv_sec < 0 ||
      usage.ru_stime.tv_usec < 0) {
    return false;
  }
  const auto micros = [](const timeval time) {
    return Add(Multiply(static_cast<std::uint64_t>(time.tv_sec), 1000000u),
               static_cast<std::uint64_t>(time.tv_usec));
  };
  value = Multiply(Add(micros(usage.ru_utime), micros(usage.ru_stime)), 1000u);
  return true;
}

[[nodiscard]] constexpr std::string_view Name(const Direction value) noexcept {
  switch (value) {
  case Direction::Disabled:
    return "disabled";
  case Direction::Basic:
    return "basic";
  case Direction::Equal:
    return "equal";
  case Direction::Detail:
    return "detail";
  }
  return "equal";
}

[[nodiscard]] constexpr std::string_view Name(const Setting value) noexcept {
  switch (value) {
  case Setting::Disabled:
    return "disabled";
  case Setting::Basic:
    return "basic";
  case Setting::Detail:
    return "detail";
  }
  return "disabled";
}

[[nodiscard]] constexpr std::string_view Name(const Operation value) noexcept {
  switch (value) {
  case Operation::Live:
    return "live";
  case Operation::Record:
    return "record";
  case Operation::Replay:
    return "replay";
  case Operation::Scenario:
    return "scenario";
  }
  return "live";
}

[[nodiscard]] constexpr std::string_view Name(const Lifecycle value) noexcept {
  return value == Lifecycle::Cold ? std::string_view{"cold"}
                                  : std::string_view{"warm"};
}

[[nodiscard]] constexpr std::string_view
Name(const rund::telemetry::Source value) noexcept {
  switch (value) {
  case rund::telemetry::Source::Compute:
    return "compute";
  case rund::telemetry::Source::Replay:
    return "replay";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
Name(const rund::telemetry::Mode value) noexcept {
  switch (value) {
  case rund::telemetry::Mode::None:
    return "none";
  case rund::telemetry::Mode::Live:
    return "live";
  case rund::telemetry::Mode::Record:
    return "record";
  case rund::telemetry::Mode::Replay:
    return "replay";
  case rund::telemetry::Mode::Scenario:
    return "scenario";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view
Name(const rund::telemetry::Preparation value) noexcept {
  switch (value) {
  case rund::telemetry::Preparation::None:
    return "none";
  case rund::telemetry::Preparation::Built:
    return "built";
  case rund::telemetry::Preparation::Reused:
    return "reused";
  }
  return "invalid";
}

[[nodiscard]] constexpr rund::telemetry::Mode
Mode(const Operation operation) noexcept {
  switch (operation) {
  case Operation::Live:
    return rund::telemetry::Mode::Live;
  case Operation::Record:
    return rund::telemetry::Mode::Record;
  case Operation::Replay:
    return rund::telemetry::Mode::Replay;
  case Operation::Scenario:
    return rund::telemetry::Mode::Scenario;
  }
  return rund::telemetry::Mode::None;
}

[[nodiscard]] constexpr bool NeedsExpected(const Operation operation) noexcept {
  return operation == Operation::Replay || operation == Operation::Scenario;
}

[[nodiscard]] constexpr rund::telemetry::Preparation
ExpectedPreparation(const Operation operation,
                    const bool expected_prepared) noexcept {
  if (operation == Operation::Live) {
    return rund::telemetry::Preparation::None;
  }
  if (operation == Operation::Record || !expected_prepared) {
    return rund::telemetry::Preparation::Built;
  }
  return rund::telemetry::Preparation::Reused;
}

static_assert(ExpectedPreparation(Operation::Live, false) ==
                  rund::telemetry::Preparation::None &&
              ExpectedPreparation(Operation::Live, true) ==
                  rund::telemetry::Preparation::None &&
              ExpectedPreparation(Operation::Record, false) ==
                  rund::telemetry::Preparation::Built &&
              ExpectedPreparation(Operation::Record, true) ==
                  rund::telemetry::Preparation::Built &&
              ExpectedPreparation(Operation::Replay, false) ==
                  rund::telemetry::Preparation::Built &&
              ExpectedPreparation(Operation::Replay, true) ==
                  rund::telemetry::Preparation::Reused &&
              ExpectedPreparation(Operation::Scenario, false) ==
                  rund::telemetry::Preparation::Built &&
              ExpectedPreparation(Operation::Scenario, true) ==
                  rund::telemetry::Preparation::Reused);

[[nodiscard]] constexpr Delta
Difference(const std::uint64_t left, const std::uint64_t right,
           const Direction left_direction = Direction::Basic,
           const Direction right_direction = Direction::Detail) noexcept {
  if (right > left) {
    return {.direction = right_direction,
            .magnitude = right - left,
            .negative = false};
  }
  if (left > right) {
    return {.direction = left_direction,
            .magnitude = left - right,
            .negative = true};
  }
  return {};
}

[[nodiscard]] constexpr bool Less(const Delta left,
                                  const Delta right) noexcept {
  const auto rank = [](const Delta value) {
    return value.direction == Direction::Equal ? 1u : value.negative ? 0u : 2u;
  };
  if (rank(left) != rank(right)) {
    return rank(left) < rank(right);
  }
  return left.negative ? left.magnitude > right.magnitude
                       : left.magnitude < right.magnitude;
}

[[nodiscard]] constexpr Half Average(const std::uint64_t left,
                                     const std::uint64_t right) noexcept {
  const std::uint64_t whole = left / 2u + right / 2u;
  const std::uint64_t remainder = left % 2u + right % 2u;
  return {.direction = Direction::Detail,
          .whole = whole + remainder / 2u,
          .half = remainder % 2u != 0u};
}

[[nodiscard]] constexpr Half Average(const Delta left,
                                     const Delta right) noexcept {
  if (left.direction == right.direction) {
    Half result = Average(left.magnitude, right.magnitude);
    result.direction = left.magnitude == 0u ? Direction::Equal : left.direction;
    return result;
  }
  if (!left.negative && !right.negative) {
    Half result = Average(left.magnitude, right.magnitude);
    result.direction = result.whole == 0u && !result.half   ? Direction::Equal
                       : left.direction == Direction::Equal ? right.direction
                                                            : left.direction;
    return result;
  }
  if (left.negative && right.negative) {
    Half result = Average(left.magnitude, right.magnitude);
    result.direction = result.whole == 0u && !result.half   ? Direction::Equal
                       : left.direction == Direction::Equal ? right.direction
                                                            : left.direction;
    return result;
  }
  const Delta negative = left.negative ? left : right;
  const Delta positive = left.negative ? right : left;
  const std::uint64_t magnitude = negative.magnitude > positive.magnitude
                                      ? negative.magnitude - positive.magnitude
                                      : positive.magnitude - negative.magnitude;
  Half result = Average(0u, magnitude);
  result.direction = magnitude == 0u ? Direction::Equal
                     : negative.magnitude > positive.magnitude
                         ? negative.direction
                         : positive.direction;
  return result;
}

constexpr Delta kNegative = Difference(11u, 5u);
constexpr Delta kPositive = Difference(10u, 15u);
constexpr Half kCrossMedian = Average(kNegative, kPositive);
constexpr Half kWideMedian =
    Average(std::numeric_limits<std::uint64_t>::max() - 1u,
            std::numeric_limits<std::uint64_t>::max());
static_assert(kNegative.direction == Direction::Basic &&
              kNegative.magnitude == 6u && kNegative.negative);
static_assert(kPositive.direction == Direction::Detail &&
              kPositive.magnitude == 5u && !kPositive.negative);
static_assert(Less(kNegative, Delta{}) && Less(Delta{}, kPositive));
static_assert(kCrossMedian.direction == Direction::Basic &&
              kCrossMedian.whole == 0u && kCrossMedian.half);
static_assert(kWideMedian.whole ==
                  std::numeric_limits<std::uint64_t>::max() - 1u &&
              kWideMedian.half);
static_assert(kPairs % 2u == 0u &&
              (95u * kPairs + 99u) / 100u - 1u == kPairs - 1u);


} // namespace rund::measure::telemetry
