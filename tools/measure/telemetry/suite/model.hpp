#pragma once

#include <rund/replay.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace allocation {
void Start() noexcept;
[[nodiscard]] std::uint64_t Stop() noexcept;
} // namespace allocation

namespace rund::measure::telemetry {

using Clock = std::chrono::steady_clock;
using Level = rund::telemetry::Level;

constexpr std::size_t kPairs = 12u;
constexpr std::size_t kWarmups = 2u;
constexpr std::size_t kMetrics = 7u;
constexpr std::size_t kOperations = 4u;
constexpr std::size_t kSettings = 3u;
constexpr std::size_t kPayloadBytes = 32u;
constexpr std::uint64_t kSequence = 1u;
constexpr rund::replay::Input kInput{.id = 1u, .schema = 1u};
constexpr std::uint64_t kSession = 0x74656c656d657472ull;

enum class Setting : std::uint8_t {
  Disabled,
  Basic,
  Detail,
};

enum class Operation : std::uint8_t {
  Live,
  Record,
  Replay,
  Scenario,
};

enum class Lifecycle : std::uint8_t {
  Cold,
  Warm,
};

enum class Direction : std::uint8_t {
  Disabled,
  Basic,
  Equal,
  Detail,
};

struct Delta final {
  Direction direction = Direction::Equal;
  std::uint64_t magnitude = 0u;
  bool negative = false;
};

struct Half final {
  Direction direction = Direction::Equal;
  std::uint64_t whole = 0u;
  bool half = false;
};

struct ReplayFields final {
  std::uint64_t input_rows = 0u;
  std::uint64_t input_bytes = 0u;
  std::uint64_t produced_rows = 0u;
  std::uint64_t choices = 0u;
  std::uint64_t evidence_rows = 0u;
  std::uint64_t evidence_bytes = 0u;
  std::uint64_t retained_bytes = 0u;
  std::uint64_t copied_bytes = 0u;
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocated_bytes = 0u;
  std::uint64_t reserved_bytes = 0u;
  std::uint64_t storage_growths = 0u;
  std::uint64_t result_hash = 0u;

  friend bool operator==(const ReplayFields &, const ReplayFields &) = default;
};

struct Counters final {
  std::uint64_t inputs = 0u;
  std::uint64_t observations = 0u;
  std::uint64_t host_events = 0u;
  std::uint64_t trace_records = 0u;
  std::uint64_t captures = 0u;
  std::uint64_t capture_hash = 0u;
  std::uint64_t logical_bytes = 0u;
  std::uint64_t encoded_bytes = 0u;
  std::uint64_t retained_bytes = 0u;
  std::uint64_t copied_bytes = 0u;
  std::uint64_t cached_bytes = 0u;
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocated_bytes = 0u;
  std::uint64_t reserved_bytes = 0u;
  std::uint64_t growths = 0u;
  std::uint64_t chunks = 0u;
  std::uint64_t segments = 0u;
  std::uint64_t cache_hits = 0u;
  std::uint64_t cache_misses = 0u;
  std::uint64_t cache_evictions = 0u;
  std::uint64_t capture_bytes = 0u;
  std::uint64_t capture_records = 0u;
  std::uint64_t capture_evicted = 0u;
  std::uint64_t capture_dropped = 0u;

  friend bool operator==(const Counters &, const Counters &) = default;
};

struct Semantics final {
  std::uint64_t status = 0u;
  std::uint64_t ordering = 0u;
  std::uint64_t transcript = 0u;
  std::uint64_t result = 0u;
  Counters counters{};
  ReplayFields replay{};

  friend bool operator==(const Semantics &, const Semantics &) = default;
};

struct Telemetry final {
  rund::telemetry::Source source = rund::telemetry::Source::Compute;
  std::uint64_t session = 0u;
  std::uint64_t scope = 0u;
  std::uint64_t code = 0u;
  rund::telemetry::Mode mode = rund::telemetry::Mode::None;
  rund::telemetry::Preparation plan = rund::telemetry::Preparation::None;
  ReplayFields replay{};

  friend bool operator==(const Telemetry &, const Telemetry &) = default;
};

struct Sample final {
  bool ok = false;
  std::uint64_t wall = 0u;
  std::uint64_t cpu = 0u;
  std::uint64_t allocations = 0u;
  rund::telemetry::Detail phases{};
  Semantics semantics{};
  Telemetry telemetry{};
};

struct Group final {
  std::array<Sample, kPairs> disabled{};
  std::array<Sample, kPairs> basic{};
  std::array<Sample, kPairs> detail{};
};

struct Suite final {
  std::array<Group, kOperations> cold{};
  std::array<Group, kOperations> warm{};
};

struct Observer final {
  std::uint32_t count = 0u;
  rund::telemetry::Event event{};

  void reset() noexcept {
    count = 0u;
    event = {};
  }

  void operator()(const rund::telemetry::Event &value) noexcept {
    ++count;
    event = value;
  }
};

struct Lane final {
  Observer observer{};
  rund::Session session{};
  std::optional<rund::replay::Record> expected{};
  bool opened = false;
};

[[nodiscard]] rund::SessionConfig Config(Observer &observer, Setting setting);
[[nodiscard]] std::array<std::byte, kPayloadBytes> Payload() noexcept;
[[nodiscard]] ReplayFields Formula(const Counters &counter, Operation operation) noexcept;
[[nodiscard]] Semantics Public(const rund::replay::Record &record, Operation operation, rund::replay::Code code) noexcept;
[[nodiscard]] Semantics Public(const rund::replay::Live &result) noexcept;
[[nodiscard]] bool Observe(Sample &sample, const Observer &observer, Setting setting, Operation operation, rund::telemetry::Preparation preparation) noexcept;
[[nodiscard]] Sample Scope(rund::Session &session, Observer &observer, Setting setting, Operation operation, rund::telemetry::Preparation preparation, const rund::replay::Record *expected, std::span<const rund::replay::Choice> choices);
[[nodiscard]] std::optional<rund::replay::Record> Load(std::string_view encoded);
[[nodiscard]] Sample Cold(Setting setting, Operation operation, std::string_view encoded, std::span<const rund::replay::Choice> choices, bool measured);
[[nodiscard]] Sample Warm(Lane &lane, Setting setting, Operation operation, std::span<const rund::replay::Choice> choices, bool expected_prepared, bool measured);
[[nodiscard]] std::optional<std::string> Seed();
[[nodiscard]] std::array<Setting, kSettings> SettingOrder(std::size_t pair) noexcept;
[[nodiscard]] std::array<Operation, kOperations> OperationOrder(std::size_t pair) noexcept;
[[nodiscard]] Sample &Pick(Group &group, Setting setting, std::size_t pair) noexcept;
[[nodiscard]] const Sample &Pick(const Group &group, Setting setting, std::size_t pair) noexcept;
[[nodiscard]] Group &GroupOf(Suite &suite, Lifecycle lifecycle, Operation operation) noexcept;
[[nodiscard]] const Group &GroupOf(const Suite &suite, Lifecycle lifecycle, Operation operation) noexcept;
[[nodiscard]] std::size_t LaneIndex(Operation operation, Setting setting) noexcept;
[[nodiscard]] bool Parity(const Group &group, std::size_t pair) noexcept;
[[nodiscard]] std::uint64_t Metric(const Sample &sample, std::size_t metric) noexcept;
[[nodiscard]] std::string_view MetricName(std::size_t metric) noexcept;
[[nodiscard]] std::string_view MetricUnit(std::size_t metric) noexcept;
void Print(const Suite &suite);
[[nodiscard]] bool Identity(std::string_view value) noexcept;
[[nodiscard]] int Run(std::string_view manifest, std::string_view artifact);

} // namespace rund::measure::telemetry
