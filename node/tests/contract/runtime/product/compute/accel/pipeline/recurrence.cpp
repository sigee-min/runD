#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>

namespace runtime_compute_pipeline_accel_detail {
namespace {

template <class T>
int CheckFixedRecurrence(rund::compute::Device &device,
                         const std::array<T, 4u> &seed, const T factor,
                         const char *const name) {
  using namespace rund::compute;
  constexpr std::size_t iterations = 17u;
  auto body = on(device)
                  .template map<T>(name, seed.size(),
                                   capture(
                                       [](auto value, auto scale) {
                                         return quantize<T>(value * scale);
                                       },
                                       factor))
                  .compile();
  auto initial = device.upload<T>(seed);
  auto serial_first = device.buffer<T>(seed.size());
  auto serial_second = device.buffer<T>(seed.size());
  auto recurrent = device.buffer<T>(seed.size());
  if (!body || !initial || !serial_first || !serial_second || !recurrent) {
    return 1;
  }

  std::array<T, 4u> serial{};
  for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
    const auto &source =
        iteration == 0u
            ? *initial
            : ((iteration & 1u) != 0u ? *serial_first : *serial_second);
    auto &target = (iteration & 1u) == 0u ? *serial_first : *serial_second;
    auto result = body->run(source, target);
    if (!result || (iteration + 1u == iterations &&
                    !result->read(target, std::span<T>{serial}))) {
      return 2;
    }
  }

  auto prepared = pipeline(device)
                      .template repeat<iterations>(*body, read(*initial),
                                                   write_final(*recurrent))
                      .prepare();
  std::array<T, 4u> actual{};
  if (!prepared) {
    std::fprintf(stderr, "%s prepare failed: %.*s\n", name,
                 static_cast<int>(prepared.error().size()),
                 prepared.error().data());
    return 3;
  }
  const Status status = prepared->run();
  if (!status) {
    std::fprintf(stderr, "%s run failed: %.*s\n", name,
                 static_cast<int>(status.error().size()),
                 status.error().data());
    return 4;
  }
  const Stats run_stats = prepared->stats();
  if (!prepared->read(*recurrent, actual) || actual != serial) {
    std::fprintf(stderr, "%s parity failed actual=%lld serial=%lld\n", name,
                 static_cast<long long>(actual[0].raw()),
                 static_cast<long long>(serial[0].raw()));
    return 5;
  }
  const Stats observed_stats = prepared->stats();
  const std::uint64_t expected_read_submits =
      run_stats.backend == Backend::Vulkan ? 1u : 0u;
  if (observed_stats.command_submits != 1u ||
      observed_stats.transfer_submissions.device_to_host !=
          expected_read_submits ||
      observed_stats.dispatches != 1u ||
      observed_stats.pipeline.step_count != 1u ||
      observed_stats.pipeline.verified_step_count != 1u) {
    std::fprintf(
        stderr, "%s topology failed submits=%llu dispatches=%llu\n", name,
        static_cast<unsigned long long>(observed_stats.command_submits),
        static_cast<unsigned long long>(observed_stats.dispatches));
    std::fprintf(
        stderr,
        "%s pre-read submits=%llu dispatches=%llu reads=%llu bytes=%llu\n",
        name, static_cast<unsigned long long>(run_stats.command_submits),
        static_cast<unsigned long long>(run_stats.dispatches),
        static_cast<unsigned long long>(run_stats.download_events),
        static_cast<unsigned long long>(run_stats.downloaded_bytes));
    return 6;
  }
  return 0;
}

} // namespace

int CheckFixedRecurrences(rund::compute::Device &device) {
  const std::array<rund::compute::Fixed<16, 16>, 4u> seed32{
      rund::compute::Fixed<16, 16>::from_raw(65537),
      rund::compute::Fixed<16, 16>::from_raw(98305),
      rund::compute::Fixed<16, 16>::from_raw(-65537),
      rund::compute::Fixed<16, 16>::from_raw(
          std::numeric_limits<std::int32_t>::max() - 2)};
  constexpr std::int64_t one64 = std::int64_t{1} << 44u;
  const std::array<rund::compute::Fixed<20, 44>, 4u> seed64{
      rund::compute::Fixed<20, 44>::from_raw(one64 + 1),
      rund::compute::Fixed<20, 44>::from_raw(one64 + (one64 >> 1u) + 1),
      rund::compute::Fixed<20, 44>::from_raw(-(one64 + 1)),
      rund::compute::Fixed<20, 44>::from_raw(
          std::numeric_limits<std::int64_t>::max() - 2)};
  if (const int fixed = CheckFixedRecurrence(
          device, seed32, rund::compute::Fixed<16, 16>::from_raw(98305),
          "pipeline-fixed-i16-f16-recurrence");
      fixed != 0) {
    return 80 + fixed;
  }
  if (const int fixed = CheckFixedRecurrence(
          device, seed64,
          rund::compute::Fixed<20, 44>::from_raw(one64 + (one64 >> 1u) + 1),
          "pipeline-fixed-i20-f44-recurrence");
      fixed != 0) {
    return 90 + fixed;
  }
  return 0;
}

} // namespace runtime_compute_pipeline_accel_detail
