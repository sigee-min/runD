#include "math.hpp"

namespace package_compute {

int Math() {
  auto product = rund::compute::on(rund::compute::Target::cpu(), left)
                     .matrix<2u, 2u>()
                     .matmul<2u, 2u>(right)
                     .collect();
  if (!product) {
    return product.exit_code();
  }
  if (*product != std::vector<rund::compute::Fixed<1, 31>>{quarter, quarter,
                                                           quarter, quarter}) {
    return FlowMismatch(__LINE__);
  }

  auto transformed =
      rund::compute::on(rund::compute::Target::cpu(), transform_real)
          .complex(transform_imag)
          .fourier()
          .real()
          .collect();
  if (!transformed) {
    return transformed.exit_code();
  }
  if (*transformed !=
      std::vector<rund::compute::Fixed<1, 31>>{half, half, half, half}) {
    return FlowMismatch(__LINE__);
  }

  auto complex_flow =
      rund::compute::on(rund::compute::Target::cpu(), transform_real)
          .complex(transform_imag)
          .fourier()
          .collect();
  if (!complex_flow) {
    return complex_flow.exit_code();
  }
  if (std::get<0>(*complex_flow) !=
          std::vector<rund::compute::Fixed<1, 31>>{half, half, half, half} ||
      std::get<1>(*complex_flow) !=
          std::vector<rund::compute::Fixed<1, 31>>{zero, zero, zero, zero}) {
    return FlowMismatch(__LINE__);
  }

  auto complex_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<rund::compute::Fixed<1, 31>>(
              "complex-copy", transform_real.size(),
              [](auto value) {
                return rund::compute::quantize<
                    rund::compute::Fixed<1, 31>,
                    rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Saturate,
                    rund::compute::Approximation::Deterministic>(value);
              })
          .complex()
          .fourier()
          .compile();
  if (!complex_program) {
    std::fprintf(stderr, "package complex program: %.*s\n",
                 static_cast<int>(complex_program.error().size()),
                 complex_program.error().data());
    return complex_program.exit_code();
  }
  auto complex_job = complex_program->resident(transform_real, transform_imag);
  if (!complex_job) {
    return complex_job.exit_code();
  }
  const auto complex_run = complex_job->run();
  if (!complex_run) {
    return complex_run.exit_code();
  }
  auto complex_outputs = complex_job->read_all();
  if (!complex_outputs) {
    return complex_outputs.exit_code();
  }
  if (std::get<0>(*complex_outputs) !=
      std::vector<rund::compute::Fixed<1, 31>>{half, half, half, half}) {
    return FlowMismatch(__LINE__);
  }

  auto static_solve = rund::compute::on(rund::compute::Target::cpu(), left)
                          .matrix<2u, 2u>()
                          .lu()
                          .solve<2u>(right)
                          .values()
                          .collect();
  if (!static_solve) {
    return static_solve.exit_code();
  }
  if (static_solve->size() != left.size()) {
    return FlowMismatch(__LINE__);
  }
  auto static_vectors = rund::compute::on(rund::compute::Target::cpu(), left)
                            .matrix<2u, 2u>()
                            .svd<rund::compute::SpectrumVectors::Thin>()
                            .vectors()
                            .collect();
  if (!static_vectors) {
    return static_vectors.exit_code();
  }
  if (static_vectors->size() != left.size()) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
