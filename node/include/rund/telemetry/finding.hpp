#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::telemetry {

enum class Cost : std::uint8_t {
  Allocation,
  Copy,
  Scan,
  Queue,
  CriticalPath,
};

enum class Unit : std::uint8_t {
  Events,
  Bytes,
  Entries,
  Nanoseconds,
};

enum class Accuracy : std::uint8_t {
  Exact,
  Unavailable,
  Saturated,
};

enum class Reference : std::uint8_t {
  None,
  ReuseEvents,
  RetainedBytes,
  QueueCapacity,
  PhaseTotal,
};

// Cause and Action are masks because a measured critical path may have exact
// ties. Preserving every argmax member is deterministic and avoids inventing a
// tie breaker.
enum class Cause : std::uint16_t {
  None = 0u,
  BufferAllocation = 1u << 0u,
  StorageGrowth = 1u << 1u,
  BoundaryCopy = 1u << 2u,
  ReplayCopy = 1u << 3u,
  GraphRead = 1u << 4u,
  QueueAtBound = 1u << 5u,
  TimingUnavailable = 1u << 6u,
  Prepare = 1u << 7u,
  Work = 1u << 8u,
  Finish = 1u << 9u,
  SubmitOverhead = 1u << 10u,
};

enum class Action : std::uint16_t {
  None = 0u,
  ReuseJob = 1u << 0u,
  ConfigureStorage = 1u << 1u,
  KeepResident = 1u << 2u,
  ReduceGraphBound = 1u << 3u,
  ReduceFanout = 1u << 4u,
  EnableDetail = 1u << 5u,
  ReuseProgram = 1u << 6u,
  ReadSelectedOutput = 1u << 7u,
  ReuseReplayPlan = 1u << 8u,
  ReduceReplayEvidence = 1u << 9u,
  BatchJobs = 1u << 10u,
};

namespace detail {

template <class> struct MemberSet;

template <> struct MemberSet<Cause> final {
  inline static constexpr std::array Values{
      Cause::BufferAllocation,
      Cause::StorageGrowth,
      Cause::BoundaryCopy,
      Cause::ReplayCopy,
      Cause::GraphRead,
      Cause::QueueAtBound,
      Cause::TimingUnavailable,
      Cause::Prepare,
      Cause::Work,
      Cause::Finish,
      Cause::SubmitOverhead,
  };
};

template <> struct MemberSet<Action> final {
  inline static constexpr std::array Values{
      Action::ReuseJob,        Action::ConfigureStorage,
      Action::KeepResident,    Action::ReduceGraphBound,
      Action::ReduceFanout,    Action::EnableDetail,
      Action::ReuseProgram,    Action::ReadSelectedOutput,
      Action::ReuseReplayPlan, Action::ReduceReplayEvidence,
      Action::BatchJobs,
  };
};

} // namespace detail

[[nodiscard]] constexpr Cause operator|(const Cause left,
                                        const Cause right) noexcept {
  return static_cast<Cause>(static_cast<std::uint16_t>(left) |
                            static_cast<std::uint16_t>(right));
}

constexpr Cause &operator|=(Cause &left, const Cause right) noexcept {
  left = left | right;
  return left;
}

[[nodiscard]] constexpr Action operator|(const Action left,
                                         const Action right) noexcept {
  return static_cast<Action>(static_cast<std::uint16_t>(left) |
                             static_cast<std::uint16_t>(right));
}

constexpr Action &operator|=(Action &left, const Action right) noexcept {
  left = left | right;
  return left;
}

[[nodiscard]] constexpr bool contains(const Cause value,
                                      const Cause member) noexcept {
  return member != Cause::None && (static_cast<std::uint16_t>(value) &
                                   static_cast<std::uint16_t>(member)) ==
                                      static_cast<std::uint16_t>(member);
}

[[nodiscard]] constexpr bool contains(const Action value,
                                      const Action member) noexcept {
  return member != Action::None && (static_cast<std::uint16_t>(value) &
                                    static_cast<std::uint16_t>(member)) ==
                                       static_cast<std::uint16_t>(member);
}

struct Finding final {
  Cost cost = Cost::Allocation;
  Unit unit = Unit::Events;
  Accuracy accuracy = Accuracy::Exact;
  Accuracy reference_accuracy = Accuracy::Unavailable;
  Reference reference_kind = Reference::None;
  Cause cause = Cause::None;
  Action action = Action::None;
  std::uint64_t observed = 0u;
  std::uint64_t reference = 0u;

  [[nodiscard]] constexpr bool exact() const noexcept {
    return accuracy == Accuracy::Exact;
  }

  [[nodiscard]] constexpr bool has_reference() const noexcept {
    return reference_kind != Reference::None &&
           reference_accuracy != Accuracy::Unavailable;
  }

  [[nodiscard]] constexpr bool
  operator==(const Finding &) const noexcept = default;
};

struct Event;

class Findings final {
public:
  static constexpr std::size_t Capacity = 5u;

  [[nodiscard]] constexpr std::size_t size() const noexcept { return count_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return count_ == 0u; }
  [[nodiscard]] constexpr const Finding &
  operator[](const std::size_t index) const noexcept {
    return values_[index];
  }
  [[nodiscard]] constexpr const Finding *begin() const noexcept {
    return values_.data();
  }
  [[nodiscard]] constexpr const Finding *end() const noexcept {
    return values_.data() + count_;
  }
  [[nodiscard]] constexpr std::span<const Finding> values() const noexcept {
    return {values_.data(), count_};
  }

private:
  constexpr bool append(const Finding finding) noexcept {
    if (count_ == Capacity) {
      return false;
    }
    values_[count_++] = finding;
    return true;
  }

  std::array<Finding, Capacity> values_{};
  std::uint8_t count_ = 0u;

  friend struct Event;
};

template <class Value> class Members;

[[nodiscard]] constexpr Members<Cause> members(Cause mask) noexcept;
[[nodiscard]] constexpr Members<Action> members(Action mask) noexcept;

template <class Value> class Members final {
public:
  static constexpr std::size_t Capacity =
      detail::MemberSet<Value>::Values.size();

  [[nodiscard]] constexpr std::size_t size() const noexcept { return count_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return count_ == 0u; }
  [[nodiscard]] constexpr const Value *begin() const noexcept {
    return values_.data();
  }
  [[nodiscard]] constexpr const Value *end() const noexcept {
    return values_.data() + count_;
  }
  [[nodiscard]] constexpr const Value &
  operator[](const std::size_t index) const noexcept {
    return values_[index];
  }

private:
  constexpr void append(const Value value) noexcept {
    values_[count_++] = value;
  }

  std::array<Value, Capacity> values_{};
  std::uint8_t count_ = 0u;

  friend constexpr Members<Cause> members(Cause mask) noexcept;
  friend constexpr Members<Action> members(Action mask) noexcept;
};

[[nodiscard]] constexpr Members<Cause> members(const Cause mask) noexcept {
  Members<Cause> out{};
  for (const Cause value : detail::MemberSet<Cause>::Values) {
    if (contains(mask, value)) {
      out.append(value);
    }
  }
  return out;
}

[[nodiscard]] constexpr Members<Action> members(const Action mask) noexcept {
  Members<Action> out{};
  for (const Action value : detail::MemberSet<Action>::Values) {
    if (contains(mask, value)) {
      out.append(value);
    }
  }
  return out;
}

[[nodiscard]] constexpr Accuracy accuracy(const std::uint64_t value) noexcept {
  return value == std::numeric_limits<std::uint64_t>::max()
             ? Accuracy::Saturated
             : Accuracy::Exact;
}

[[nodiscard]] std::string_view name(Cost value) noexcept;
[[nodiscard]] std::string_view name(Unit value) noexcept;
[[nodiscard]] std::string_view name(Accuracy value) noexcept;
[[nodiscard]] std::string_view name(Reference value) noexcept;
[[nodiscard]] std::string_view name(Cause value) noexcept;
[[nodiscard]] std::string_view name(Action value) noexcept;

template <class Writer>
  requires requires(Writer &writer, const std::string_view text) {
    { writer(text) } -> std::same_as<void>;
  }
void describe(const Finding &finding, Writer &&writer) noexcept(
    noexcept(std::declval<Writer &>()(std::declval<std::string_view>()))) {
  Writer &out = writer;
  out(name(finding.cost));
  out(" cause=");
  const Members<Cause> causes = members(finding.cause);
  if (causes.empty()) {
    out(name(Cause::None));
  } else {
    bool separator = false;
    for (const Cause cause : causes) {
      if (separator) {
        out(",");
      }
      out(name(cause));
      separator = true;
    }
  }
  out(" action=");
  const Members<Action> actions = members(finding.action);
  if (actions.empty()) {
    out(name(Action::None));
  } else {
    bool separator = false;
    for (const Action action : actions) {
      if (separator) {
        out(",");
      }
      out(name(action));
      separator = true;
    }
  }
}

static_assert(std::is_trivially_copyable_v<Finding>);
static_assert(std::is_trivially_copyable_v<Findings>);
static_assert(std::is_trivially_copyable_v<Members<Cause>>);
static_assert(std::is_trivially_copyable_v<Members<Action>>);
static_assert(Members<Cause>::Capacity == 11u);
static_assert(Members<Action>::Capacity == 11u);

} // namespace rund::telemetry
