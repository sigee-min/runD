#include "core.hpp"

namespace rund::measure::compute {

template <class ProgramResult, class Validate, class... Input>
bool MeasureWorkload(const std::string_view family,
                     const std::string_view variant, const Backend backend,
                     const std::size_t input_count,
                     const std::size_t active_count,
                     const std::size_t iterations, ProgramResult &program,
                     Validate validate, const Input &...input) {
  if (!program) {
    std::fprintf(stderr, "workload %s/%.*s/%.*s compile failed: %.*s\n",
                 Name(backend), static_cast<int>(family.size()), family.data(),
                 static_cast<int>(variant.size()), variant.data(),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = program->resident(input...);
  if (!job) {
    std::fprintf(stderr, "workload %s/%.*s/%.*s resident failed: %.*s\n",
                 Name(backend), static_cast<int>(family.size()), family.data(),
                 static_cast<int>(variant.size()), variant.data(),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  const auto warmup = job->run();
  HashEvidence warmup_evidence{};
  if (!warmup ||
      !CaptureOutput(*job, backend, family, warmup_evidence, validate)) {
    std::fprintf(stderr, "workload %s/%.*s/%.*s warmup validation failed\n",
                 Name(backend), static_cast<int>(family.size()), family.data(),
                 static_cast<int>(variant.size()), variant.data());
    return false;
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  WarmCounters warm{};
  for (std::size_t index = 0u; index < iterations; ++index) {
    const auto begin = Clock::now();
    const auto status = job->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "workload %s/%.*s/%.*s run failed: %.*s\n",
                   Name(backend), static_cast<int>(family.size()),
                   family.data(), static_cast<int>(variant.size()),
                   variant.data(), static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    warm.observe(job->stats());
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double median_us = Median(samples);
  const double input_rate =
      median_us == 0.0 ? 0.0
                       : static_cast<double>(input_count) * 1.0e6 / median_us;
  const double active_rate =
      median_us == 0.0 ? 0.0
                       : static_cast<double>(active_count) * 1.0e6 / median_us;
  HashEvidence evidence{};
  if (!CaptureOutput(*job, backend, family, evidence, validate)) {
    return false;
  }
  const bool reference_ok = CheckReference(
      backend, ReferenceKey{family, variant, input_count}, evidence);
  const auto stats = job->stats();
  const auto memory = job->memory();
  std::printf("workload,%s,%.*s,%.*s,%s,%zu,%zu,%zu,%.3f,%.3f,%.3f",
              Name(backend), static_cast<int>(family.size()), family.data(),
              static_cast<int>(variant.size()), variant.data(),
              reference_ok ? "ok" : "reference_failed", input_count,
              active_count, iterations, median_us, input_rate, active_rate);
  PrintStats(stats);
  PrintWarm(warm);
  std::printf(",%llu,%llu\n",
              static_cast<unsigned long long>(memory.resident.current),
              static_cast<unsigned long long>(memory.staging.current));
  return warm.zero() && reference_ok;
}

bool SparseWorkloads(const Backend backend, const std::size_t count,
                     const std::size_t iterations) {
  const std::size_t active = count / 16u;
  std::vector<std::uint32_t> input(count);
  for (std::size_t index = 0u; index < count; ++index) {
    input[index] = static_cast<std::uint32_t>(count - index);
  }
  const auto expensive = [](auto value) {
    const auto a = (value * 1664525u + 1013904223u) ^
                   rund::compute::shr_logical<7u>(value);
    const auto b = (a * 22695477u + 1u) ^ rund::compute::shr_logical<9u>(a);
    const auto c =
        (b * 1103515245u + 12345u) ^ rund::compute::shr_logical<11u>(b);
    const auto d =
        (c * 214013u + 2531011u) ^ rund::compute::shr_logical<13u>(c);
    return (d * 134775813u + 1u) ^ rund::compute::shr_logical<15u>(d);
  };
  auto dense_map =
      rund::compute::on(TargetFor(backend))
          .map<std::uint32_t>("measure-dense-expensive", count, expensive)
          .compile();
  auto sparse_map = rund::compute::on(TargetFor(backend))
                        .map<std::uint32_t>("measure-filter-source", count,
                                            [](auto value) { return value; })
                        .filter([](auto value) { return (value & 15u) == 0u; })
                        .map("measure-filter-expensive", expensive)
                        .compile();
  auto dense_sort = rund::compute::on(TargetFor(backend))
                        .map<std::uint32_t>("measure-dense-sort", count,
                                            [](auto value) { return value; })
                        .sort()
                        .compile();
  auto sparse_sort = rund::compute::on(TargetFor(backend))
                         .map<std::uint32_t>("measure-filter-sort", count,
                                             [](auto value) { return value; })
                         .filter([](auto value) { return (value & 15u) == 0u; })
                         .sort()
                         .compile();

  const auto has_count = [](const std::size_t expected) {
    return [expected](const auto &values) { return values.size() == expected; };
  };
  const auto is_sorted = [](const std::size_t expected) {
    return [expected](const auto &values) {
      return values.size() == expected &&
             std::is_sorted(values.begin(), values.end());
    };
  };
  bool ok = true;
  ok = MeasureWorkload("map", "dense_expensive", backend, count, count,
                       iterations, dense_map, has_count(count), input) &&
       ok;
  ok = MeasureWorkload("map", "stable_filter_1_of_16_expensive", backend, count,
                       active, iterations, sparse_map, has_count(active),
                       input) &&
       ok;
  ok = MeasureWorkload("sort", "dense", backend, count, count, iterations,
                       dense_sort, is_sorted(count), input) &&
       ok;
  ok = MeasureWorkload("sort", "stable_filter_1_of_16", backend, count, active,
                       iterations, sparse_sort, is_sorted(active), input) &&
       ok;
  return ok;
}

bool CollectiveWorkloads(const Backend backend, const std::size_t count,
                         const std::size_t iterations) {
  constexpr std::size_t segment_width = 256u;
  std::vector<std::uint32_t> input(count, 1u);
  std::vector<std::uint32_t> heads(count, 0u);
  for (std::size_t index = 0u; index < count; index += segment_width) {
    heads[index] = 1u;
  }
  auto scan = rund::compute::on(TargetFor(backend))
                  .map<std::uint32_t>("measure-scan", count,
                                      [](auto value) { return value; })
                  .scan(rund::compute::Scan::InclusiveSum)
                  .compile();
  auto reduce = rund::compute::on(TargetFor(backend))
                    .map<std::uint32_t>("measure-reduce", count,
                                        [](auto value) { return value; })
                    .reduce(rund::compute::Reduce::Sum)
                    .compile();
  auto segmented_scan =
      rund::compute::on(TargetFor(backend))
          .map<std::uint32_t>("measure-segmented-scan", count,
                              [](auto value) { return value; })
          .segmented_scan(count, rund::compute::Scan::InclusiveSum)
          .compile();
  auto segmented_reduce =
      rund::compute::on(TargetFor(backend))
          .map<std::uint32_t>("measure-segmented-reduce", count,
                              [](auto value) { return value; })
          .segmented_reduce(count, rund::compute::Reduce::Sum)
          .compile();

  const auto scan_valid = [count](const auto &values) {
    return values.size() == count && !values.empty() && values.back() == count;
  };
  const auto reduce_valid = [count](const auto &values) {
    return values.size() == 1u && values[0] == count;
  };
  const auto segmented_scan_valid = [count](const auto &values) {
    return values.size() == count && !values.empty() &&
           values.back() == segment_width;
  };
  const auto segmented_reduce_valid = [count](const auto &values) {
    if (values.size() != count) {
      return false;
    }
    const auto segments = count / segment_width;
    return std::all_of(values.begin(), values.begin() + segments,
                       [](const auto value) { return value == segment_width; });
  };
  bool ok = true;
  ok = MeasureWorkload("scan", "inclusive_sum", backend, count, count,
                       iterations, scan, scan_valid, input) &&
       ok;
  ok = MeasureWorkload("reduce", "sum", backend, count, count, iterations,
                       reduce, reduce_valid, input) &&
       ok;
  ok = MeasureWorkload("segmented_scan", "inclusive_sum_width_256", backend,
                       count, count, iterations, segmented_scan,
                       segmented_scan_valid, input, heads) &&
       ok;
  ok = MeasureWorkload("segmented_reduce", "sum_width_256", backend, count,
                       count, iterations, segmented_reduce,
                       segmented_reduce_valid, input, heads) &&
       ok;
  return ok;
}

#if !defined(RUND_COMPUTE_FOCUS)
template <class T>
bool FixedWidening(const Backend backend, const char *const label) {
  constexpr std::size_t count = 4096u;
  constexpr std::size_t iterations = 11u;
  using Raw = typename T::Raw;
  std::vector<T> input(count);
  constexpr unsigned shift = T::fraction_bits - 8u;
  for (std::size_t index = 0u; index < count; ++index) {
    input[index] = T::from_raw(
        static_cast<Raw>((static_cast<Raw>(index % 63u) + Raw{1}) << shift));
  }

  const auto compile_begin = Clock::now();
  auto program =
      rund::compute::on(TargetFor(backend))
          .template map<T>(label, count,
                           [](auto value) {
                             return rund::compute::quantize<T>(value * value);
                           })
          .compile();
  const auto compile_end = Clock::now();
  const double compile_us =
      std::chrono::duration<double, std::micro>(compile_end - compile_begin)
          .count();
  std::printf("fixed_cold,%s,%s,%zu,%.3f\n", Name(backend), label, count,
              compile_us);
  const auto valid = [count](const auto &values) {
    return values.size() == count;
  };
  return MeasureWorkload("fixed_map", label, backend, count, count, iterations,
                         program, valid, input);
}

bool FixedWidening32(const Backend backend) {
  return FixedWidening<Fixed<16, 16>>(backend, "fixed_i16_f16_widen");
}

bool FixedWidening64(const Backend backend) {
  return FixedWidening<Fixed<20, 44>>(backend, "fixed_i20_f44_widen");
}
#endif


} // namespace rund::measure::compute
