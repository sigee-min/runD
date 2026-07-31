#include "contract/program/compute/dsl/ops/local.hpp"
#include <kernel/program/compute/cpu.hpp>
namespace program_compute_contract {
namespace {

[[nodiscard]] rund::compute_dsl::ComputeValue
Store(const rund::compute_dsl::ComputeValue value) noexcept {
  return rund::compute_dsl::quantize<
      1u, 31u, rund::kernel::ComputeRounding::NearestEven,
      rund::kernel::ComputeOverflow::Saturate,
      rund::kernel::ComputeApproximation::Deterministic>(value);
}
using namespace dsl_support;
[[nodiscard]] auto BuildOverloadSurfaceOp() {
  i32 ax[4]{}, ay[4]{}, az[4]{}, bx[4]{}, by[4]{}, bz[4]{}, out[4]{};
  auto body = rund::compute_dsl::bind(4u)
                  .fixed<1, 31>()
                  .read<"ax">(ax)
                  .read<"ay">(ay)
                  .read<"az">(az)
                  .read<"bx">(bx)
                  .read<"by">(by)
                  .read<"bz">(bz)
                  .write<"out">(out);
  return rund::compute_dsl::def("dsl-overload-surface")
      .on(body)
      .map([](auto i, auto b) {
        auto ax = b.template read<"ax">();
        auto ay = b.template read<"ay">();
        auto az = b.template read<"az">();
        auto bx = b.template read<"bx">();
        auto by = b.template read<"by">();
        auto bz = b.template read<"bz">();
        auto out = b.template write<"out">();
        using Axis = rund::compute_dsl::Axis;
        using AngleOp = rund::compute_dsl::AngleOp;
        using MatOp = rund::compute_dsl::MatOp;
        using MetricOp = rund::compute_dsl::MetricOp;
        using Norm = rund::compute_dsl::Norm;
        out[i] =
            Store(
                rund::compute_dsl::dot(ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::dot(ax[i], ay[i], az[i], bx[i], by[i],
                                       bz[i]) +
                rund::compute_dsl::len(ax[i], ay[i]) +
                rund::compute_dsl::len(ax[i], ay[i], az[i]) +
                rund::compute_dsl::len(MetricOp::Squared, ax[i], ay[i]) +
                rund::compute_dsl::len(MetricOp::Squared, ax[i], ay[i], az[i]) +
                rund::compute_dsl::dist(ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::dist(ax[i], ay[i], az[i], bx[i], by[i],
                                        bz[i])) +
            Store(
                rund::compute_dsl::dist(MetricOp::Squared, ax[i], ay[i], bx[i],
                                        by[i]) +
                rund::compute_dsl::dist(MetricOp::Squared, ax[i], ay[i], az[i],
                                        bx[i], by[i], bz[i]) +
                rund::compute_dsl::len(Norm::L1, ax[i], ay[i]) +
                rund::compute_dsl::len(Norm::L1, ax[i], ay[i], az[i]) +
                rund::compute_dsl::len(Norm::LInf, ax[i], ay[i]) +
                rund::compute_dsl::len(Norm::LInf, ax[i], ay[i], az[i]) +
                rund::compute_dsl::dist(Norm::L1, ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::dist(Norm::L1, ax[i], ay[i], az[i], bx[i],
                                        by[i], bz[i])) +
            Store(
                rund::compute_dsl::dist(Norm::LInf, ax[i], ay[i], bx[i],
                                        by[i]) +
                rund::compute_dsl::dist(Norm::LInf, ax[i], ay[i], az[i], bx[i],
                                        by[i], bz[i]) +
                rund::compute_dsl::angle(AngleOp::Cosine, ax[i], ay[i], bx[i],
                                         by[i]) +
                rund::compute_dsl::angle(AngleOp::Cosine, ax[i], ay[i], az[i],
                                         bx[i], by[i], bz[i]) +
                rund::compute_dsl::sum(ax[i], ay[i]) +
                rund::compute_dsl::sum(ax[i], ay[i], az[i]) +
                rund::compute_dsl::sum(ax[i], ay[i], az[i], bx[i]) +
                rund::compute_dsl::sum(rund::compute_dsl::SumOp::Abs, ax[i],
                                       ay[i])) +
            Store(
                rund::compute_dsl::sum(rund::compute_dsl::SumOp::Abs, ax[i],
                                       ay[i], az[i]) +
                rund::compute_dsl::sum(rund::compute_dsl::SumOp::Squared, ax[i],
                                       ay[i]) +
                rund::compute_dsl::sum(rund::compute_dsl::SumOp::Squared, ax[i],
                                       ay[i], az[i]) +
                rund::compute_dsl::diff(ax[i], ay[i]) +
                rund::compute_dsl::diff(ax[i], ay[i], az[i]) +
                rund::compute_dsl::absdiff(ax[i], ay[i]) +
                rund::compute_dsl::diff(
                    rund::compute_dsl::DifferenceOrder::Second, ax[i], ay[i],
                    az[i]) +
                rund::compute_dsl::diff(
                    rund::compute_dsl::DifferenceOrder::Third, ax[i], ay[i],
                    az[i], bx[i])) +
            Store(
                rund::compute_dsl::mean(ax[i], ay[i]) +
                rund::compute_dsl::mean(ax[i], ay[i], az[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Abs, ax[i],
                                        ay[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Abs, ax[i],
                                        ay[i], az[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Squared,
                                        ax[i], ay[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Squared,
                                        ax[i], ay[i], az[i]) +
                rund::compute_dsl::centered(rund::compute_dsl::CenteredOp::Abs,
                                            ax[i], ay[i]) +
                rund::compute_dsl::centered(
                    rund::compute_dsl::CenteredOp::Squared, ax[i], ay[i])) +
            Store(
                rund::compute_dsl::var(ax[i], ay[i]) +
                rund::compute_dsl::var(ax[i], ay[i], az[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Cubic,
                                        ax[i], ay[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Cubic,
                                        ax[i], ay[i], az[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Quartic,
                                        ax[i], ay[i]) +
                rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Quartic,
                                        ax[i], ay[i], az[i]) +
                rund::compute_dsl::mean(
                    rund::compute_dsl::StandardizedOp::Cubic, ax[i], ay[i]) +
                rund::compute_dsl::mean(
                    rund::compute_dsl::StandardizedOp::Cubic, ax[i], ay[i],
                    az[i])) +
            Store(
                rund::compute_dsl::mean(
                    rund::compute_dsl::StandardizedOp::Quartic, ax[i], ay[i]) +
                rund::compute_dsl::mean(
                    rund::compute_dsl::StandardizedOp::Quartic, ax[i], ay[i],
                    az[i]) +
                rund::compute_dsl::rms(ax[i], ay[i]) +
                rund::compute_dsl::rms(ax[i], ay[i], az[i]) +
                rund::compute_dsl::weighted_mean(ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::weighted_mean(ax[i], ay[i], az[i], bx[i],
                                                 by[i], bz[i]) +
                rund::compute_dsl::ratio(ax[i], ay[i]) +
                rund::compute_dsl::proportion(ax[i], ay[i])) +
            Store(
                rund::compute_dsl::proportion(Axis::X, ax[i], ay[i]) +
                rund::compute_dsl::proportion(Axis::Z, ax[i], ay[i], az[i]) +
                rund::compute_dsl::zscore(ax[i], ay[i], az[i]) +
                rund::compute_dsl::near(ax[i], ay[i]) +
                rund::compute_dsl::smootherstep(ax[i], ay[i], az[i]) +
                rund::compute_dsl::bandpass(ax[i], ay[i], az[i]) +
                rund::compute_dsl::bandstop(ax[i], ay[i], az[i]) +
                rund::compute_dsl::softsign(ax[i])) +
            Store(
                rund::compute_dsl::softsign(ax[i], ay[i]) +
                rund::compute_dsl::saturate(rund::compute_dsl::mean(
                    rund::compute_dsl::softsign(ax[i]),
                    rund::compute_dsl::fixed_max(ax[i]))) +
                rund::compute_dsl::saturate(rund::compute_dsl::mean(
                    rund::compute_dsl::softsign(ax[i], ay[i]),
                    rund::compute_dsl::fixed_max(ax[i]))) +
                rund::compute_dsl::activation(
                    rund::compute_dsl::ActivationOp::Relu, ax[i]) +
                rund::compute_dsl::activation(
                    rund::compute_dsl::ActivationOp::Relu, ax[i], ay[i]) +
                rund::compute_dsl::activation(
                    rund::compute_dsl::ActivationOp::LeakyRelu, ax[i], ay[i]) +
                rund::compute_dsl::activation(
                    rund::compute_dsl::ActivationOp::HardSigmoid, ax[i]) +
                rund::compute_dsl::activation(
                    rund::compute_dsl::ActivationOp::HardSwish, ax[i])) +
            Store(
                rund::compute_dsl::activation(
                    rund::compute_dsl::ActivationOp::HardTanh, ax[i]) +
                rund::compute_dsl::window(
                    rund::compute_dsl::WindowOp::Parabolic, ax[i]) +
                rund::compute_dsl::window(
                    rund::compute_dsl::WindowOp::Triangular, ax[i]) +
                rund::compute_dsl::window(rund::compute_dsl::WindowOp::Hann,
                                          ax[i]) +
                rund::compute_dsl::window(rund::compute_dsl::WindowOp::Hamming,
                                          ax[i]) +
                rund::compute_dsl::window(rund::compute_dsl::WindowOp::Blackman,
                                          ax[i]) +
                rund::compute_dsl::window(rund::compute_dsl::WindowOp::Lanczos,
                                          ax[i]) +
                rund::compute_dsl::sin(ax[i])) +
            Store(
                rund::compute_dsl::cos(ax[i]) + rund::compute_dsl::tan(ax[i]) +
                rund::compute_dsl::exp(ax[i]) + rund::compute_dsl::log(ax[i]) +
                rund::compute_dsl::pow(ax[i], ay[i]) +
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Conj,
                                           rund::compute_dsl::ComplexPart::Real,
                                           ax[i], ay[i]) +
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Conj,
                                           rund::compute_dsl::ComplexPart::Imag,
                                           ax[i], ay[i]) +
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Mul,
                                           rund::compute_dsl::ComplexPart::Real,
                                           ax[i], ay[i], bx[i], by[i])) +
            Store(
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Mul,
                                           rund::compute_dsl::ComplexPart::Imag,
                                           ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Abs,
                                           ax[i], ay[i]) +
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Abs2,
                                           ax[i], ay[i]) +
                rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Phase,
                                           ax[i], ay[i]) +
                rund::compute_dsl::huber(ax[i], ay[i]) +
                rund::compute_dsl::positive_part(ax[i]) +
                rund::compute_dsl::negative_part(ax[i]) +
                rund::compute_dsl::keep_if(ax[i], ay[i])) +
            Store(
                rund::compute_dsl::zero_if(ax[i], ay[i]) +
                rund::compute_dsl::bary(Axis::X, ax[i], ay[i], bx[i], by[i],
                                        az[i], bz[i], ay[i], bz[i]) +
                rund::compute_dsl::bary(Axis::Y, ax[i], ay[i], bx[i], by[i],
                                        az[i], bz[i], ay[i], bz[i]) +
                rund::compute_dsl::bary(Axis::Z, ax[i], ay[i], bx[i], by[i],
                                        az[i], bz[i], ay[i], bz[i]) +
                rund::compute_dsl::poly_deriv(ax[i], ay[i], az[i]) +
                rund::compute_dsl::poly_deriv(ax[i], ay[i], az[i], bx[i]) +
                rund::compute_dsl::reflect(Axis::X, ax[i], ay[i], bx[i],
                                           by[i]) +
                rund::compute_dsl::reflect(Axis::Y, ax[i], ay[i], bx[i],
                                           by[i])) +
            Store(
                rund::compute_dsl::reflect(Axis::X, ax[i], ay[i], az[i], bx[i],
                                           by[i], bz[i]) +
                rund::compute_dsl::reflect(Axis::Y, ax[i], ay[i], az[i], bx[i],
                                           by[i], bz[i]) +
                rund::compute_dsl::reflect(Axis::Z, ax[i], ay[i], az[i], bx[i],
                                           by[i], bz[i]) +
                rund::compute_dsl::mat(MatOp::Trace, ax[i], ay[i]) +
                rund::compute_dsl::mat(MatOp::Trace, ax[i], ay[i], az[i]) +
                rund::compute_dsl::mat(MatOp::Determinant, ax[i], ay[i], bx[i],
                                       by[i]) +
                rund::compute_dsl::mat(MatOp::Transpose, Axis::X, ax[i], ay[i],
                                       bx[i], by[i]) +
                rund::compute_dsl::mat(MatOp::Transpose, Axis::Y, ax[i], ay[i],
                                       bx[i], by[i])) +
            Store(
                rund::compute_dsl::mat(MatOp::Transpose, Axis::X, ax[i], ay[i],
                                       az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::mat(MatOp::Transpose, Axis::Y, ax[i], ay[i],
                                       az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::mat(MatOp::Transpose, Axis::Z, ax[i], ay[i],
                                       az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::all(ax[i], ay[i], az[i]) +
                rund::compute_dsl::any(ax[i], ay[i], az[i]));
      });
}

} // namespace

int RunComputeDslOpsOverloadContract() {
  const auto op = BuildOverloadSurfaceOp();
  TEST_ASSERT(op.ok());
  TEST_ASSERT(op.ir().ok);
  TEST_ASSERT(!op.ir().canonical_bytes.empty());
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(artifact.ok);
  return 0;
}
} // namespace program_compute_contract
