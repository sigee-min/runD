#include "local.hpp"

#include <rund/telemetry/event.hpp>

#include "src/compute/memory/profile.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rund_node_test_compute_reuse {
namespace {

using TelemetryProfile = rund::compute::telemetry::Profile;
using TelemetryRate = rund::compute::telemetry::Rate;
using TelemetryShare = rund::compute::telemetry::Share;

constexpr TelemetryRate kMissingRate{};
constexpr TelemetryRate kExactRate{.numerator = 14u, .denominator = 4u};
constexpr TelemetryRate kSaturatedRate{
    .numerator = std::numeric_limits<std::uint64_t>::max(), .denominator = 4u};
constexpr TelemetryShare kMissingShare{};
constexpr TelemetryShare kWideShare{
    .selected = std::numeric_limits<std::uint64_t>::max() - 1u,
    .other = std::numeric_limits<std::uint64_t>::max() - 1u};

static_assert(!kMissingRate.available() && !kMissingRate.value().has_value());
static_assert(kExactRate.available() && kExactRate.value() == 3.5L);
static_assert(kSaturatedRate.saturated() && !kSaturatedRate.available() &&
              !kSaturatedRate.value().has_value());
static_assert(!kMissingShare.available() && !kMissingShare.value().has_value());
static_assert(kWideShare.available() && kWideShare.value() == 0.5L);
static_assert(!std::is_default_constructible_v<TelemetryProfile>);
static_assert(!std::is_aggregate_v<TelemetryProfile>);
static_assert(std::is_nothrow_move_constructible_v<TelemetryProfile>);

template <class T, std::size_t Count>
[[nodiscard]] bool CheckHostIdentity(rund::compute::Device &device,
                                     const std::string_view name,
                                     const std::array<T, Count> &input) {
  auto program =
      rund::compute::on(device)
          .template map<T>(name, Count, [](auto value) { return value; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "host identity compile failed name=%.*s reason=%.*s\n",
                 static_cast<int>(name.size()), name.data(),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto output = program->run(std::span<const T>{input});
  const std::vector<T> expected{input.begin(), input.end()};
  const bool aligned =
      !output || output->empty() ||
      reinterpret_cast<std::uintptr_t>(output->data()) % alignof(T) == 0u;
  if (!output || *output != expected || !aligned) {
    std::fprintf(
        stderr, "host identity failed name=%.*s read=%d count=%zu aligned=%d\n",
        static_cast<int>(name.size()), name.data(), output ? 1 : 0,
        output ? output->size() : 0u, aligned ? 1 : 0);
    return false;
  }
  return true;
}

} // namespace

[[nodiscard]] bool CheckTelemetryMath() {
  using namespace rund::compute;
  using namespace rund::compute::telemetry;

  const Stats zero{.command_capacity = 8u,
                   .command_inflight_peak = 5u,
                   .original_dispatches = 9u,
                   .final_dispatches = 4u,
                   .kernel_samples = 1u};
  const Profile zero_profile = detail::ProfileAccess::make(
      std::make_shared<const DeviceInfo>(
          DeviceInfo{.name = "test", .driver = "test", .driver_details = ""}),
      zero, {});
  const Focus zero_focus = zero_profile.largest_time();
  if (!zero_focus.available() || zero_focus.nanoseconds() != 0u ||
      zero_focus.saturated() || !zero_focus.includes(Stage::Kernel) ||
      zero_focus.includes(Stage::ShaderCompile) ||
      zero_profile.dispatch_reduction() != Rate{5u, 9u} ||
      zero_profile.command_pressure() != Rate{5u, 8u}) {
    return false;
  }

  constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
  const Stats tied{.kernel_ns = limit,
                   .kernel_samples = 1u,
                   .shader_compile_ns = limit,
                   .descriptor_setup_ns = 17u,
                   .readback_ns = limit};
  const Profile tied_profile = detail::ProfileAccess::make(
      std::make_shared<const DeviceInfo>(
          DeviceInfo{.name = "test", .driver = "test", .driver_details = ""}),
      tied, {});
  const Focus tied_focus = tied_profile.largest_time();
  return tied_focus.available() && tied_focus.saturated() &&
         tied_focus.nanoseconds() == limit &&
         tied_focus.includes(Stage::ShaderCompile) &&
         tied_focus.includes(Stage::Kernel) &&
         tied_focus.includes(Stage::Readback) &&
         !tied_focus.includes(Stage::SpirvCompile) &&
         !tied_focus.includes(Stage::PipelineCreate) &&
         !tied_focus.includes(Stage::DescriptorSetup) &&
         !tied_focus.includes(Stage::SubmitWait);
}

static_assert(sizeof(rund::compute::Run) == 1392u);
static_assert(sizeof(rund::compute::Result<rund::compute::Run>) == 1400u);
static_assert(sizeof(rund::compute::ResidencyStats) == 232u);
static_assert(sizeof(rund::compute::PipelineStats) == 416u);
static_assert(sizeof(rund::compute::Stats) == 896u);
static_assert(sizeof(rund::compute::MemoryStats) == 288u);
static_assert(sizeof(rund::compute::telemetry::Profile) == 1200u);
static_assert(sizeof(rund::telemetry::Event) == 304u);
static_assert(rund::compute::Stats{}.pipeline.preparation_evidence ==
              rund::compute::PreparationEvidenceSource::Unavailable);
static_assert(rund::compute::Stats{}.pipeline.sealed_repetition_count == 0u);
static_assert(rund::compute::Stats{}.pipeline.coalesced_repetition_count == 0u);
static_assert(alignof(rund::compute::Run) == alignof(std::uint64_t));
static_assert(std::is_nothrow_copy_constructible_v<rund::compute::Run>);
static_assert(std::is_nothrow_move_constructible_v<rund::compute::Run>);

[[nodiscard]] bool CheckHostIdentities(rund::compute::Device &device) {
  using Q16_16 = rund::compute::Fixed<16u, 16u>;
  using Q20_44 = rund::compute::Fixed<20u, 44u>;
  return CheckHostIdentity(device, "host-i32",
                           std::array<std::int32_t, 3>{-7, 0, 11}) &&
         CheckHostIdentity(device, "host-u32",
                           std::array<std::uint32_t, 3>{0u, 7u, 11u}) &&
         CheckHostIdentity(device, "host-i64",
                           std::array<std::int64_t, 3>{-7, 0, 11}) &&
         CheckHostIdentity(device, "host-u64",
                           std::array<std::uint64_t, 3>{0u, 7u, 11u}) &&
         CheckHostIdentity(device, "host-q16-16",
                           std::array<Q16_16, 3>{Q16_16::from_raw(-7),
                                                 Q16_16::zero(),
                                                 Q16_16::from_raw(11)}) &&
         CheckHostIdentity(device, "host-q20-44",
                           std::array<Q20_44, 3>{Q20_44::from_raw(-7),
                                                 Q20_44::zero(),
                                                 Q20_44::from_raw(11)}) &&
         CheckHostIdentity(device, "host-empty", std::array<std::int32_t, 0>{});
}

} // namespace rund_node_test_compute_reuse
