#include "core.hpp"

namespace rund::measure::telemetry {

[[nodiscard]] std::array<Setting, kSettings>
SettingOrder(const std::size_t pair) noexcept {
  constexpr std::array<std::array<Setting, kSettings>, 6u> orders{{
      {Setting::Disabled, Setting::Basic, Setting::Detail},
      {Setting::Disabled, Setting::Detail, Setting::Basic},
      {Setting::Basic, Setting::Detail, Setting::Disabled},
      {Setting::Detail, Setting::Basic, Setting::Disabled},
      {Setting::Basic, Setting::Disabled, Setting::Detail},
      {Setting::Detail, Setting::Disabled, Setting::Basic},
  }};
  return orders[pair % orders.size()];
}

[[nodiscard]] std::array<Operation, kOperations>
OperationOrder(const std::size_t pair) noexcept {
  constexpr std::array<std::array<Operation, kOperations>, 4u> orders{{
      {Operation::Live, Operation::Record, Operation::Scenario,
       Operation::Replay},
      {Operation::Record, Operation::Replay, Operation::Live,
       Operation::Scenario},
      {Operation::Replay, Operation::Scenario, Operation::Record,
       Operation::Live},
      {Operation::Scenario, Operation::Live, Operation::Replay,
       Operation::Record},
  }};
  return orders[pair % orders.size()];
}

[[nodiscard]] Sample &Pick(Group &group, const Setting setting,
                                     const std::size_t pair) noexcept {
  switch (setting) {
  case Setting::Disabled:
    return group.disabled[pair];
  case Setting::Basic:
    return group.basic[pair];
  case Setting::Detail:
    return group.detail[pair];
  }
  return group.disabled[pair];
}

[[nodiscard]] const Sample &Pick(const Group &group,
                                           const Setting setting,
                                           const std::size_t pair) noexcept {
  switch (setting) {
  case Setting::Disabled:
    return group.disabled[pair];
  case Setting::Basic:
    return group.basic[pair];
  case Setting::Detail:
    return group.detail[pair];
  }
  return group.disabled[pair];
}

[[nodiscard]] Group &GroupOf(Suite &suite, const Lifecycle lifecycle,
                                       const Operation operation) noexcept {
  return lifecycle == Lifecycle::Cold ? suite.cold[Index(operation)]
                                      : suite.warm[Index(operation)];
}

[[nodiscard]] const Group &
GroupOf(const Suite &suite, const Lifecycle lifecycle,
        const Operation operation) noexcept {
  return lifecycle == Lifecycle::Cold ? suite.cold[Index(operation)]
                                      : suite.warm[Index(operation)];
}

[[nodiscard]] std::size_t LaneIndex(const Operation operation,
                                              const Setting setting) noexcept {
  return Index(operation) * kSettings + Index(setting);
}

[[nodiscard]] bool Parity(const Group &group, const std::size_t pair) noexcept {
  const Sample &disabled = group.disabled[pair];
  const Sample &basic = group.basic[pair];
  const Sample &detail = group.detail[pair];
  return disabled.ok && basic.ok && detail.ok &&
         disabled.semantics == basic.semantics &&
         basic.semantics == detail.semantics &&
         basic.telemetry == detail.telemetry;
}

[[nodiscard]] std::uint64_t
Metric(const Sample &sample, const std::size_t metric) noexcept {
  switch (metric) {
  case 0u:
    return sample.wall;
  case 1u:
    return sample.cpu;
  case 2u:
    return sample.allocations;
  case 3u:
    return sample.semantics.replay.copied_bytes;
  case 4u:
    return sample.phases.prepare_ns;
  case 5u:
    return sample.phases.work_ns;
  default:
    return sample.phases.finish_ns;
  }
}

[[nodiscard]] std::string_view
MetricName(const std::size_t metric) noexcept {
  constexpr std::array names{"wall",    "cpu",  "allocations", "copied",
                             "prepare", "work", "finish"};
  return names[metric];
}

[[nodiscard]] std::string_view
MetricUnit(const std::size_t metric) noexcept {
  constexpr std::array units{"ns", "ns", "count", "bytes", "ns", "ns", "ns"};
  return units[metric];
}


} // namespace rund::measure::telemetry
