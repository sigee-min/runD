#include "model.hpp"

namespace package_compute {

int Record() {
  auto branched = rund::compute::on(rund::compute::Target::cpu(), values)
                      .branch([](auto stage) {
                        const auto fields = rund::compute::record(
                            stage, stage.reduce(rund::compute::Reduce::Sum));
                        const auto adjusted = fields.template get<0>().combine(
                            "package-record", fields.template get<1>(),
                            [](auto value, auto sum) { return value + sum; });
                        return rund::compute::outputs(fields, adjusted);
                      })
                      .collect();
  if (!branched) {
    std::fprintf(stderr, "package branch: %.*s\n",
                 static_cast<int>(branched.error().size()),
                 branched.error().data());
    return branched.exit_code();
  }
  if (std::get<0>(std::get<0>(*branched)) !=
          std::vector<std::uint32_t>{3u, 1u, 4u, 2u} ||
      std::get<1>(std::get<0>(*branched)) != 10u ||
      std::get<1>(*branched) !=
          std::vector<std::uint32_t>{13u, 11u, 14u, 12u}) {
    const auto &source = std::get<0>(std::get<0>(*branched));
    const auto &adjusted = std::get<1>(*branched);
    std::fprintf(stderr,
                 "package branch: sum=%u source=%u,%u,%u,%u "
                 "adjusted=%u,%u,%u,%u\n",
                 std::get<1>(std::get<0>(*branched)), source[0], source[1],
                 source[2], source[3], adjusted[0], adjusted[1], adjusted[2],
                 adjusted[3]);
    return FlowMismatch(__LINE__);
  }

  auto combined =
      rund::compute::on(rund::compute::Target::cpu(), values)
          .combine("package-combine", combine_side,
                   [](auto left, auto right) { return left + right; })
          .collect();
  if (!combined) {
    return combined.exit_code();
  }
  if (*combined != std::vector<std::uint32_t>{4u, 3u, 7u, 6u}) {
    return FlowMismatch(__LINE__);
  }

  const std::array<std::int32_t, 4u> typed_values{1, 2, 3, 4};
  auto typed_record =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .input<std::int32_t>(typed_values.size())
          .zip_input<std::uint32_t>(combine_side.size())
          .map("package-typed-record",
               [](auto value, auto weight) {
                 return rund::compute::record(
                     rund::compute::field<PackageValue>(value),
                     rund::compute::field<PackageWeight>(weight),
                     rund::compute::field<PackageScore>(value * weight));
               })
          .branch([](auto rows) {
            const auto zipped =
                rund::compute::zip(rows.template get<PackageValue>(),
                                   rows.template get<PackageWeight>())
                    .map("package-typed-zip", [](auto value, auto weight) {
                      return value * weight;
                    });
            return rund::compute::outputs(rows.template get<PackageScore>(),
                                          zipped);
          })
          .compile();
  if (!typed_record) {
    return typed_record.exit_code();
  }
  auto typed_record_job = typed_record->resident(typed_values, combine_side);
  if (!typed_record_job) {
    return typed_record_job.exit_code();
  }
  const auto typed_record_run = typed_record_job->run();
  if (!typed_record_run) {
    return typed_record_run.exit_code();
  }
  auto typed_record_output = typed_record_job->read_all();
  if (!typed_record_output) {
    return typed_record_output.exit_code();
  }
  if (std::get<0>(*typed_record_output) !=
          std::vector<std::uint32_t>{1u, 4u, 9u, 16u} ||
      std::get<1>(*typed_record_output) !=
          std::vector<std::uint32_t>{1u, 4u, 9u, 16u}) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
