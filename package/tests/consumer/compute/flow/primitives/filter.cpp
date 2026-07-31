#include "model.hpp"

namespace package_compute {

int Filter() {
  auto filter_output =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .filter([](auto value) { return value > 2; })
          .collect();
  if (!filter_output) {
    return filter_output.exit_code();
  }
  if (*filter_output != std::vector<std::int64_t>{3, 4}) {
    return FlowMismatch(__LINE__);
  }
  auto filter_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int64_t>("filter-copy", filter_values.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .compile();
  if (!filter_program) {
    return filter_program.exit_code();
  }
  auto filter_job = filter_program->resident(filter_values);
  if (!filter_job) {
    return filter_job.exit_code();
  }
  const auto filter_run = filter_job->run();
  if (!filter_run) {
    return filter_run.exit_code();
  }
  auto resident_filter = filter_job->read();
  if (!resident_filter) {
    return resident_filter.exit_code();
  }
  if (*resident_filter != std::vector<std::int64_t>{3, 4}) {
    return FlowMismatch(__LINE__);
  }
  auto typed_map =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .map("typed-mask",
               [](auto value) {
                 return rund::compute::select<std::uint64_t>(value > 2, 1u, 0u);
               })
          .collect();
  if (!typed_map) {
    return typed_map.exit_code();
  }
  if (*typed_map != std::vector<std::uint64_t>{0u, 0u, 1u, 1u}) {
    return FlowMismatch(__LINE__);
  }
  auto narrow_mask =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .map("package-mask",
               [](auto value) { return rund::compute::mask(value > 2); })
          .collect();
  if (!narrow_mask) {
    return narrow_mask.exit_code();
  }
  if (*narrow_mask != std::vector<std::uint32_t>{0u, 0u, 1u, 1u}) {
    return FlowMismatch(__LINE__);
  }
  auto filter_count =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .filter([](auto value) { return value > 2; })
          .count()
          .collect();
  if (!filter_count) {
    return filter_count.exit_code();
  }
  if (*filter_count != std::vector<std::uint64_t>{2u}) {
    return FlowMismatch(__LINE__);
  }
  auto bounded_sum =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .pipe([](auto stage) {
            return stage.filter([](auto value) { return value > 2; });
          })
          .pipe([](auto stage) {
            return stage.reduce(rund::compute::Reduce::Sum);
          })
          .pipe([](auto scalar) {
            return scalar.map("package-pipe-scalar",
                              [](auto value) { return value + 1; });
          })
          .collect();
  if (!bounded_sum) {
    return bounded_sum.exit_code();
  }
  if (*bounded_sum != std::vector<std::int64_t>{8}) {
    return FlowMismatch(__LINE__);
  }
  auto bounded_scan =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .filter([](auto value) { return value > 2; })
          .scan(rund::compute::Scan::InclusiveSum)
          .collect();
  if (!bounded_scan) {
    return bounded_scan.exit_code();
  }
  if (*bounded_scan != std::vector<std::int64_t>{3, 7}) {
    return FlowMismatch(__LINE__);
  }
  auto bounded_sort = rund::compute::on(rund::compute::Target::cpu(), values)
                          .filter([](auto value) { return value > 1u; })
                          .sort()
                          .collect();
  if (!bounded_sort) {
    return bounded_sort.exit_code();
  }
  if (*bounded_sort != std::vector<std::uint32_t>{2u, 3u, 4u}) {
    return FlowMismatch(__LINE__);
  }
  auto bounded_order =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .filter([](auto value) { return value > 1; })
          .argsort()
          .collect();
  if (!bounded_order) {
    return bounded_order.exit_code();
  }
  if (*bounded_order != std::vector<std::uint32_t>{0u, 1u, 2u}) {
    return FlowMismatch(__LINE__);
  }

  const auto normalize = [](auto stage) {
    const auto total = stage.reduce(rund::compute::Reduce::Sum);
    return stage.combine("package-normalize", total,
                         [](auto value, auto sum) { return value * 90 / sum; });
  };
  auto normalized =
      rund::compute::on(rund::compute::Target::cpu(), filter_values)
          .filter([](auto value) { return value > 1; })
          .pipe(normalize)
          .collect();
  if (!normalized) {
    return normalized.exit_code();
  }
  if (*normalized != std::vector<std::int64_t>{20, 30, 40}) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
