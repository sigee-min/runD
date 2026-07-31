#pragma once

#include <rund/compute/device/info.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>
#include <rund/telemetry/finding.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace rund::compute {
namespace detail {
struct JobState;
struct ProfileAccess;
} // namespace detail

namespace telemetry {

// A rational metric over raw evidence. The numerator and denominator remain
// public so formatting never becomes a second authority. value() is available
// only while those components prove an exact value.
struct Rate final {
  std::uint64_t numerator{};
  std::uint64_t denominator{};

  [[nodiscard]] constexpr bool saturated() const noexcept {
    constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    return numerator == limit || denominator == limit;
  }

  [[nodiscard]] constexpr bool available() const noexcept {
    return denominator != 0u && !saturated();
  }

  [[nodiscard]] constexpr std::optional<long double> value() const noexcept {
    if (!available()) {
      return std::nullopt;
    }
    return static_cast<long double>(numerator) /
           static_cast<long double>(denominator);
  }

  [[nodiscard]] constexpr bool
  operator==(const Rate &) const noexcept = default;
};

// A share whose exact denominator is selected + other. Keeping both terms
// avoids overflowing a uint64_t denominator when individually exact counters
// are large.
struct Share final {
  std::uint64_t selected{};
  std::uint64_t other{};

  [[nodiscard]] constexpr bool saturated() const noexcept {
    constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    return selected == limit || other == limit;
  }

  [[nodiscard]] constexpr bool available() const noexcept {
    return (selected != 0u || other != 0u) && !saturated();
  }

  [[nodiscard]] constexpr std::optional<long double> value() const noexcept {
    if (!available()) {
      return std::nullopt;
    }
    const long double selected_value = static_cast<long double>(selected);
    return selected_value / (selected_value + static_cast<long double>(other));
  }

  [[nodiscard]] constexpr bool
  operator==(const Share &) const noexcept = default;
};

enum class Stage : std::uint8_t {
  ShaderCompile,
  SpirvCompile,
  PipelineCreate,
  DescriptorSetup,
  SubmitWait,
  Kernel,
  Readback,
};

class Focus final {
public:
  [[nodiscard]] constexpr bool available() const noexcept {
    return stages_ != 0u;
  }

  [[nodiscard]] constexpr std::uint64_t nanoseconds() const noexcept {
    return nanoseconds_;
  }

  [[nodiscard]] constexpr bool saturated() const noexcept {
    return available() &&
           nanoseconds_ == std::numeric_limits<std::uint64_t>::max();
  }

  [[nodiscard]] constexpr bool includes(const Stage stage) const noexcept {
    return (stages_ & stage_bit(stage)) != 0u;
  }

  [[nodiscard]] constexpr bool
  operator==(const Focus &) const noexcept = default;

private:
  friend class Profile;

  [[nodiscard]] static constexpr std::uint8_t
  stage_bit(const Stage stage) noexcept {
    const std::uint8_t index = static_cast<std::uint8_t>(stage);
    return index <= static_cast<std::uint8_t>(Stage::Readback)
               ? static_cast<std::uint8_t>(1u << index)
               : 0u;
  }

  constexpr void consider(const Stage stage, const std::uint64_t nanoseconds,
                          const bool measured) noexcept {
    if (!measured) {
      return;
    }
    if (!available() || nanoseconds > nanoseconds_) {
      nanoseconds_ = nanoseconds;
      stages_ = stage_bit(stage);
      return;
    }
    if (nanoseconds == nanoseconds_) {
      stages_ = static_cast<std::uint8_t>(stages_ | stage_bit(stage));
    }
  }

  std::uint64_t nanoseconds_{};
  std::uint8_t stages_{};
};

class Profile final {
public:
  Profile(const Profile &) = default;
  Profile(Profile &&) noexcept = default;
  Profile &operator=(const Profile &) = default;
  Profile &operator=(Profile &&) noexcept = default;

  [[nodiscard]] const DeviceInfo &device() const noexcept { return device_; }
  [[nodiscard]] const Stats &execution() const noexcept { return execution_; }
  [[nodiscard]] const MemoryStats &memory() const noexcept { return memory_; }

  // Allocation-free actions derived by the same telemetry owner used by
  // Session Compute events. Raw snapshots remain the only counter authority.
  [[nodiscard]] ::rund::telemetry::Findings findings() const noexcept;

  // Cumulative kernel nanoseconds per measured kernel sample.
  [[nodiscard]] constexpr Rate kernel_time() const noexcept {
    return {execution_.kernel_ns, execution_.kernel_samples};
  }

  // Logical algorithmic dispatches per recorded physical queue submission.
  // This is a literal quotient, not an encoding-efficiency estimate.
  [[nodiscard]] constexpr Rate dispatches_per_submit() const noexcept {
    return {execution_.dispatches, execution_.command_submits};
  }

  // Peak occupied slots over the immutable backend command envelope.
  // A zero capacity is unavailable evidence on backends without this owner.
  [[nodiscard]] constexpr Rate command_pressure() const noexcept {
    return {execution_.command_inflight_peak, execution_.command_capacity};
  }

  // Cache hits versus compilations admitted by this execution snapshot.
  [[nodiscard]] constexpr Share pipeline_cache() const noexcept {
    return {execution_.pipeline_cache_hits, execution_.pipeline_compiles};
  }

  // Reused versus newly allocated backend buffers.
  [[nodiscard]] constexpr Share buffer_reuse() const noexcept {
    return {execution_.buffer_reuses, execution_.buffer_allocations};
  }

  // Reused versus newly allocated descriptor sets.
  [[nodiscard]] constexpr Share descriptor_reuse() const noexcept {
    return {execution_.descriptor_reuses,
            execution_.descriptor_set_allocations};
  }

  // Removed algorithmic dispatches over the original algorithmic dispatches.
  // A malformed final count greater than the original yields a zero numerator
  // rather than wrapping unsigned arithmetic.
  [[nodiscard]] constexpr Rate dispatch_reduction() const noexcept {
    return {execution_.original_dispatches >= execution_.final_dispatches
                ? execution_.original_dispatches - execution_.final_dispatches
                : 0u,
            execution_.original_dispatches};
  }

  // Internal versus externally visible round-trip bytes.
  [[nodiscard]] constexpr Share internal_traffic() const noexcept {
    return {execution_.internal_roundtrip_bytes,
            execution_.external_roundtrip_bytes};
  }

  // Current bytes over the configured byte budget. A zero budget is explicit
  // unavailable evidence rather than an unlimited or zero-pressure guess.
  [[nodiscard]] Result<Rate> memory_usage(const MemoryCategory category) const {
    const MemoryCounter *counter = nullptr;
    switch (category) {
    case MemoryCategory::Host:
      counter = &memory_.host;
      break;
    case MemoryCategory::Frame:
      counter = &memory_.frame;
      break;
    case MemoryCategory::Tile:
      counter = &memory_.tile;
      break;
    case MemoryCategory::Resident:
      counter = &memory_.resident;
      break;
    case MemoryCategory::Staging:
      counter = &memory_.staging;
      break;
    case MemoryCategory::Device:
      counter = &memory_.device;
      break;
    case MemoryCategory::Transfer:
      counter = &memory_.transfer;
      break;
    }
    if (counter == nullptr) {
      return Result<Rate>::fail(Reason::ProfileMemoryCategoryInvalid);
    }
    return Result<Rate>::success({counter->current, counter->budget});
  }

  // Returns every timing stage tied for the largest cumulative measured
  // duration. There is no threshold, backend substitution, or tie breaker.
  [[nodiscard]] constexpr Focus largest_time() const noexcept {
    Focus focus{};
    focus.consider(Stage::ShaderCompile, execution_.shader_compile_ns,
                   execution_.shader_compile_ns != 0u);
    focus.consider(Stage::SpirvCompile, execution_.spirv_compile_ns,
                   execution_.spirv_compile_ns != 0u);
    focus.consider(Stage::PipelineCreate, execution_.pipeline_create_ns,
                   execution_.pipeline_create_ns != 0u);
    focus.consider(Stage::DescriptorSetup, execution_.descriptor_setup_ns,
                   execution_.descriptor_setup_ns != 0u);
    focus.consider(Stage::SubmitWait, execution_.submit_wait_ns,
                   execution_.submit_wait_ns != 0u);
    focus.consider(Stage::Kernel, execution_.kernel_ns,
                   execution_.kernel_timing_available());
    focus.consider(Stage::Readback, execution_.readback_ns,
                   execution_.readback_ns != 0u);
    return focus;
  }

private:
  friend struct ::rund::compute::detail::ProfileAccess;

  Profile(DeviceInfo device, const Stats execution,
          const MemoryStats memory) noexcept
      : device_(std::move(device)), execution_(execution), memory_(memory) {}

  DeviceInfo device_;
  Stats execution_{};
  MemoryStats memory_{};
};

} // namespace telemetry

namespace detail {
[[nodiscard]] Result<telemetry::Profile>
job_profile(const std::shared_ptr<JobState> &state) noexcept;
}

} // namespace rund::compute
