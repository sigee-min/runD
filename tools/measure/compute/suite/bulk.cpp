#include "core.hpp"

namespace rund::measure::compute {

#if defined(RUND_COMPUTE_FOCUS)
void PrintBulkColumns() {
  std::fputs(
      "bulk_columns,backend,family,status,count,passes,samples,median_us,"
      "logical_ops,logical_ops_per_s,value_bytes,coefficient_bytes,"
      "logical_bytes,logical_bytes_per_s,"
      "kernel_median_us,submit_wait_median_us,dispatches,command_submits,"
      "graph_hash,output_hash,warm_zero,resident_bytes,staging_bytes\n",
      stdout);
}

template <class ProgramResult, class... Input>
bool BulkBench(const std::string_view family, const Backend backend,
               const BulkCost cost, const std::size_t samples,
               ProgramResult &program, const ReferenceKey reference,
               const Input &...input) {
  if (samples == 0u) {
    return false;
  }
  if (!program) {
    std::fprintf(stderr, "bulk %s/%.*s compile failed: %.*s\n", Name(backend),
                 static_cast<int>(family.size()), family.data(),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = program->resident(input...);
  if (!job) {
    std::fprintf(stderr, "bulk %s/%.*s resident failed: %.*s\n", Name(backend),
                 static_cast<int>(family.size()), family.data(),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  if (!job->run()) {
    std::fprintf(stderr, "bulk %s/%.*s warmup failed\n", Name(backend),
                 static_cast<int>(family.size()), family.data());
    return false;
  }

  std::vector<double> wall;
  std::vector<double> kernel;
  std::vector<double> submit_wait;
  wall.reserve(samples);
  kernel.reserve(samples);
  submit_wait.reserve(samples);
  WarmCounters warm{};
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const auto begin = Clock::now();
    const auto status = job->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "bulk %s/%.*s run failed: %.*s\n", Name(backend),
                   static_cast<int>(family.size()), family.data(),
                   static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    const auto stats = job->stats();
    warm.observe(stats);
    wall.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
    kernel.push_back(static_cast<double>(stats.kernel_ns) / 1'000.0);
    submit_wait.push_back(static_cast<double>(stats.submit_wait_ns) / 1'000.0);
  }

  HashEvidence evidence{};
  if (!CaptureOutput(*job, backend, family, evidence)) {
    return false;
  }
  const bool reference_ok = CheckReference(backend, reference, evidence);
  const double median_us = Median(wall);
  const double ops_per_s =
      median_us == 0.0
          ? 0.0
          : static_cast<double>(cost.logical_ops) * 1.0e6 / median_us;
  const double bytes_per_s =
      median_us == 0.0
          ? 0.0
          : static_cast<double>(cost.logical_bytes()) * 1.0e6 / median_us;
  const auto stats = job->stats();
  const auto memory = job->memory();
  std::printf("bulk,%s,%.*s,%s,%llu,%llu,%zu,%.3f,%llu,%.3f,%llu,%llu,%llu,"
              "%.3f,%.3f,%.3f,%llu,%llu,%llu,%llu,%u,%llu,%llu\n",
              Name(backend), static_cast<int>(family.size()), family.data(),
              reference_ok ? "ok" : "reference_failed",
              static_cast<unsigned long long>(cost.count),
              static_cast<unsigned long long>(cost.passes), samples, median_us,
              static_cast<unsigned long long>(cost.logical_ops), ops_per_s,
              static_cast<unsigned long long>(cost.value_bytes),
              static_cast<unsigned long long>(cost.coefficient_bytes),
              static_cast<unsigned long long>(cost.logical_bytes()),
              bytes_per_s, Median(kernel), Median(submit_wait),
              static_cast<unsigned long long>(stats.dispatches),
              static_cast<unsigned long long>(stats.command_submits),
              static_cast<unsigned long long>(evidence.graph),
              static_cast<unsigned long long>(evidence.output),
              warm.zero() ? 1u : 0u,
              static_cast<unsigned long long>(memory.resident.current),
              static_cast<unsigned long long>(memory.staging.current));
  return warm.zero() && reference_ok;
}

bool BulkMatrix(const Backend backend, const std::size_t samples) {
  constexpr std::size_t side = 512u;
  constexpr std::size_t count = side * side;
  std::vector<std::int32_t> left(count);
  std::vector<std::int32_t> right(count);
  for (std::size_t row = 0u; row < side; ++row) {
    for (std::size_t column = 0u; column < side; ++column) {
      const std::size_t index = row * side + column;
      left[index] =
          static_cast<std::int32_t>((row * 17u + column * 13u) % 7u) - 3;
      right[index] =
          static_cast<std::int32_t>((row * 11u + column * 19u) % 7u) - 3;
    }
  }
  auto program = rund::compute::on(TargetFor(backend))
                     .map<std::int32_t>("bulk-matrix", count,
                                        [](auto value) { return value; })
                     .matrix<side, side>()
                     .matmul<side, side>()
                     .compile();
  return BulkBench("matrix", backend, MatrixBulkCost, samples, program,
                   ReferenceKey{"bulk", "matrix", count}, left, right);
}

bool BulkTransform(const Backend backend, const std::size_t samples) {
  constexpr std::size_t count = 1u << 20u;
  std::vector<Fixed<1, 31>> real(count);
  std::vector<Fixed<1, 31>> imag(count);
  for (std::size_t index = 0u; index < count; ++index) {
    const auto real_raw =
        static_cast<std::int32_t>((index * 17u + 5u) & 65'535u) - 32'768;
    const auto imag_raw =
        static_cast<std::int32_t>((index * 29u + 3u) & 65'535u) - 32'768;
    real[index] = Fixed<1, 31>::from_raw(real_raw);
    imag[index] = Fixed<1, 31>::from_raw(imag_raw);
  }
  auto program = rund::compute::on(TargetFor(backend))
                     .map<Fixed<1, 31>>(
                         "bulk-transform", count,
                         [](auto value) {
                           return rund::compute::quantize<Fixed<1, 31>>(value);
                         })
                     .complex()
                     .fourier()
                     .compile();
  return BulkBench("transform", backend, TransformBulkCost, samples, program,
                   ReferenceKey{"bulk", "transform", count}, real, imag);
}

bool Bulk(const Backend backend, const std::size_t samples) {
  bool ok = BulkMatrix(backend, samples);
  ok = BulkTransform(backend, samples) && ok;
  return ok;
}
#endif

} // namespace rund::measure::compute
