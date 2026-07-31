#pragma once

#include <rund/compute/backend.hpp>
#include <rund/compute/status.hpp>
#include <rund/replay/code.hpp>
#include <rund/telemetry/finding.hpp>

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace rund::telemetry {

enum class Source : std::uint8_t {
  Compute,
  Replay,
};

enum class Level : std::uint8_t {
  Basic,
  Detail,
};

enum class Mode : std::uint8_t {
  None,
  Live,
  Record,
  Replay,
  Scenario,
};

enum class Preparation : std::uint8_t {
  None,
  Built,
  Reused,
};

struct Compute final {
  compute::Backend backend = compute::Backend::Cpu;
  compute::Code code = compute::Code::Ok;
  std::uint64_t graph = 0u;
  std::uint32_t workers = 0u;
  std::uint32_t active_workers = 0u;
  std::uint64_t tiles = 0u;
  std::uint64_t dispatches = 0u;
  std::uint64_t command_submits = 0u;
  std::uint64_t buffer_allocations = 0u;
  std::uint64_t buffer_reuses = 0u;
  std::uint64_t copied_bytes = 0u;
  std::uint64_t graph_read_bytes = 0u;
  std::uint64_t kernel_ns = 0u;
  std::uint64_t kernel_samples = 0u;
  std::uint64_t submit_wait_ns = 0u;
};

struct Replay final {
  replay::Code code = replay::Code::Ok;
  Mode mode = Mode::None;
  Preparation plan = Preparation::None;
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
};

struct Detail final {
  std::uint64_t prepare_ns = 0u;
  std::uint64_t work_ns = 0u;
  std::uint64_t finish_ns = 0u;
};

struct Queue final {
  std::uint64_t depth = 0u;
  std::uint64_t capacity = 0u;
};

struct Event final {
  Source source = Source::Compute;
  Level level = Level::Basic;
  std::uint64_t session = 0u;
  std::uint64_t scope = 0u;
  Compute compute{};
  Replay replay{};
  Queue queue{};
  Detail detail{};

  [[nodiscard]] Findings findings() const noexcept;

  [[nodiscard]] std::string_view error() const noexcept;
};

template <class Writer>
  requires requires(Writer &writer, const std::string_view text) {
    { writer(text) } -> std::same_as<void>;
  }
void describe(const Event &event, Writer &&writer) noexcept(
    noexcept(std::declval<Writer &>()(std::declval<std::string_view>()))) {
  Writer &out = writer;
  bool separator = false;
  for (const Finding &finding : event.findings()) {
    if (separator) {
      out("\n");
    }
    describe(finding, out);
    separator = true;
  }
}

static_assert(std::is_trivially_copyable_v<Event>);

} // namespace rund::telemetry
