#include "math.hpp"

namespace package_compute {

int Program() {
  auto outputs = rund::compute::on(rund::compute::Target::cpu())
                     .map<std::uint32_t>("double", values.size(),
                                         [](auto value) { return value * 2u; })
                     .branch([](auto doubled) {
                       return rund::compute::outputs(
                           doubled, doubled.map("increment", [](auto value) {
                             return value + 1u;
                           }));
                     })
                     .compile();
  if (!outputs) {
    return outputs.exit_code();
  }
  auto output_job = outputs->resident(values);
  if (!output_job) {
    return output_job.exit_code();
  }
  const auto output_run = output_job->run();
  if (!output_run) {
    return output_run.exit_code();
  }
  auto second = output_job->template read<1u>();
  if (!second) {
    return second.exit_code();
  }
  if (*second != std::vector<std::uint32_t>{7u, 3u, 9u, 5u}) {
    return FlowMismatch(__LINE__);
  }

  auto bounded_program =
      rund::compute::on(rund::compute::Target::cpu())
          .map<std::uint32_t>("bounded-count", values.size(),
                              [](auto value) { return value; })
          .expand(
              rund::compute::MaxItems{1u}, [](auto value) { return value; },
              [](auto value, auto) { return value; })
          .compile();
  if (!bounded_program) {
    return bounded_program.exit_code();
  }
  auto bounded_job = bounded_program->resident(values);
  if (!bounded_job) {
    return bounded_job.exit_code();
  }
  const auto bounded_status = bounded_job->run();
  if (bounded_status || bounded_status.code() != rund::compute::Code::Binding ||
      bounded_status.error() != "compute_bounded_count_invalid") {
    std::fprintf(stderr,
                 "package bounded-count expected ok=0 code=%u "
                 "reason=compute_bounded_count_invalid; actual ok=%u code=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(rund::compute::Code::Binding),
                 bounded_status ? 1u : 0u,
                 static_cast<unsigned>(bounded_status.code()),
                 static_cast<int>(bounded_status.error().size()),
                 bounded_status.error().data());
    return FlowMismatch(__LINE__);
  }

  auto factor_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<rund::compute::Fixed<1, 31>>(
              "typed-lu", transform_real.size(),
              [](auto value) {
                return rund::compute::quantize<
                    rund::compute::Fixed<1, 31>,
                    rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Saturate,
                    rund::compute::Approximation::Deterministic>(value);
              })
          .matrix({2u, 2u, 1u})
          .lu()
          .compile();
  if (!factor_program) {
    return factor_program.exit_code();
  }
  auto factor_flow = rund::compute::on(rund::compute::Target::cpu(), left)
                         .matrix({2u, 2u, 1u})
                         .lu()
                         .collect();
  if (!factor_flow) {
    return factor_flow.exit_code();
  }
  if (std::get<2>(*factor_flow) != std::vector<std::uint32_t>{0u}) {
    return FlowMismatch(__LINE__);
  }
  auto solve_flow = rund::compute::on(rund::compute::Target::cpu(), left)
                        .matrix({2u, 2u, 1u})
                        .lu()
                        .solve(left, 2u)
                        .collect();
  if (!solve_flow) {
    return solve_flow.exit_code();
  }
  if (std::get<1>(*solve_flow) != std::vector<std::uint32_t>{0u}) {
    return FlowMismatch(__LINE__);
  }
  auto matrix_solve_flow = rund::compute::on(rund::compute::Target::cpu(), left)
                               .matrix<2u, 2u>()
                               .solve<rund::compute::FactorOp::Lu, 2u>(left)
                               .collect();
  if (!matrix_solve_flow) {
    return matrix_solve_flow.exit_code();
  }
  if (std::get<1>(*matrix_solve_flow) != std::vector<std::uint32_t>{0u}) {
    return FlowMismatch(__LINE__);
  }
  auto spectrum_flow = rund::compute::on(rund::compute::Target::cpu(), left)
                           .matrix({2u, 2u, 1u})
                           .svd<rund::compute::SpectrumVectors::Thin>()
                           .collect();
  if (!spectrum_flow) {
    return spectrum_flow.exit_code();
  }
  if (std::get<0>(*spectrum_flow).size() != 2u ||
      std::get<1>(*spectrum_flow).size() != 4u ||
      std::get<2>(*spectrum_flow) != std::vector<std::uint32_t>{0u}) {
    return FlowMismatch(__LINE__);
  }
  auto factor_job = factor_program->resident(left);
  if (!factor_job) {
    return factor_job.exit_code();
  }
  const auto factor_run = factor_job->run();
  if (!factor_run) {
    return factor_run.exit_code();
  }
  auto factor_status = factor_job->template read<2u>();
  if (!factor_status) {
    return factor_status.exit_code();
  }
  if (*factor_status != std::vector<std::uint32_t>{0u}) {
    return FlowMismatch(__LINE__);
  }

  auto plan = rund::compute::on(rund::compute::Target::cpu())
                  .map<std::uint32_t>("copy", values.size(),
                                      [](auto value) { return value; })
                  .gather(indices.size())
                  .compile();
  return plan ? 0 : plan.exit_code();
}

} // namespace package_compute
