#pragma once

#include "../../model.hpp"
#include "../backing.hpp"
#include "../model.hpp"
#include "schema.hpp"

#include <rund/compute/telemetry.hpp>
#include <rund/compute/virtual.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rund::measure::compute::virtual_crossover::detail {

using Pipeline = ::rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>;
using Profile = ::rund::compute::telemetry::Profile;

struct PreparedPoint final {
  Backend backend;
  std::size_t logical_count;
  std::size_t radius;
  std::shared_ptr<virtual_residency::MemoryBacking> input_backing;
  std::shared_ptr<virtual_residency::MemoryBacking> output_backing;
  Pipeline pipeline;
  ::rund::compute::PipelinePlan plan;
  std::vector<std::int32_t> observed;

  PreparedPoint(const Backend, std::size_t, std::size_t,
                std::shared_ptr<virtual_residency::MemoryBacking>,
                std::shared_ptr<virtual_residency::MemoryBacking>,
                Pipeline) noexcept;
};

struct BackendEvidence final {
  virtual_residency::WallSamples samples{};
  std::optional<Profile> profile;
  double prepare_us{};
  double cold_us{};
  double p25_us{};
  double p50_us{};
  double p75_us{};
  double p95_us{};
  double mad_us{};
  std::uint64_t output_hash{};
};

[[nodiscard]] std::size_t active_count(std::size_t, ActiveRatio) noexcept;
[[nodiscard]] std::size_t frame_elements(std::size_t) noexcept;
[[nodiscard]] std::uint64_t input_frame_bytes(std::size_t) noexcept;
[[nodiscard]] ::rund::compute::ResidencyConfig
residency_config(Backend, std::size_t) noexcept;

void seed(std::span<std::int32_t>) noexcept;
[[nodiscard]] std::int32_t seed_value(std::size_t) noexcept;

[[nodiscard]] std::unique_ptr<PreparedPoint>
prepare(Backend, std::size_t logical_count, std::size_t radius) noexcept;

[[nodiscard]] bool valid_output(std::span<const std::int32_t>, std::size_t,
                                std::size_t) noexcept;
[[nodiscard]] std::uint64_t content_hash(std::span<const std::int32_t>,
                                         std::size_t) noexcept;
[[nodiscard]] double microseconds(Clock::duration) noexcept;
[[nodiscard]] bool valid_preparation(const Profile &, Backend) noexcept;
[[nodiscard]] bool valid_warm(const PreparedPoint &, const Profile &,
                              std::size_t) noexcept;
[[nodiscard]] bool observe_output(PreparedPoint &, std::size_t) noexcept;
[[nodiscard]] bool cold_run(PreparedPoint &, std::size_t, double &) noexcept;

[[nodiscard]] bool run_abba(PreparedPoint &, PreparedPoint &, std::size_t,
                            virtual_residency::WallSamples *,
                            virtual_residency::WallSamples *, bool) noexcept;
void summarize(BackendEvidence &) noexcept;
void print_point(std::size_t, std::size_t, ActiveRatio, std::size_t,
                 const PreparedPoint &, const PreparedPoint &,
                 const BackendEvidence &, const BackendEvidence &);
[[nodiscard]] bool measure_point(std::size_t, std::size_t, ActiveRatio,
                                 std::size_t) noexcept;

} // namespace rund::measure::compute::virtual_crossover::detail
