#include "local.hpp"

namespace rund::node::test_contract::boundary_modes {

template <class T>
[[nodiscard]] bool CheckFixed(const rund::compute::Backend backend,
                              FixedEvidence &evidence) {
  using namespace rund::compute;
  auto invalid = Target(backend)
                     .template map<T>("boundary-transform-invalid", kTail,
                                      [](auto value) { return value; })
                     .complex()
                     .fourier()
                     .compile();
  if (invalid ||
      invalid.error() != "compute_transform_count_not_power_of_two") {
    return false;
  }

  std::vector<T> signal(kTail + 1u, Zero<T>());
  signal[0u] = Small<T>();
  auto transform = Target(backend)
                       .template map<T>("boundary-transform", signal.size(),
                                        [](auto value) { return value; })
                       .complex()
                       .fourier()
                       .compile();
  if (!transform) {
    return false;
  }
  auto transform_job = transform->resident(signal, signal);
  const std::vector<T> transformed(signal.size(), Small<T>());
  if (!transform_job ||
      !CheckSuccess(*transform_job, backend, "transform-boundary",
                    evidence.transform, [&](const auto &output) {
                      return std::get<0>(output) == transformed &&
                             std::get<1>(output) == transformed;
                    })) {
    return false;
  }

  std::vector<T> matrices(kTail, Small<T>());
  auto factor = Target(backend)
                    .template map<T>("boundary-factor", kTail,
                                     [](auto value) { return value; })
                    .matrix({1u, 1u, kTail})
                    .lu()
                    .compile();
  auto solve = Target(backend)
                   .template map<T>("boundary-solve", kTail,
                                    [](auto value) { return value; })
                   .matrix({1u, 1u, kTail})
                   .lu()
                   .solve(1u)
                   .compile();
  auto spectrum = Target(backend)
                      .template map<T>("boundary-spectrum", kTail,
                                       [](auto value) { return value; })
                      .matrix({1u, 1u, kTail})
                      .template svd<SpectrumVectors::Values>()
                      .compile();
  if (!factor || !solve || !spectrum) {
    return false;
  }
  auto factor_job = factor->resident(matrices);
  auto solve_job = solve->resident(matrices, matrices);
  auto spectrum_job = spectrum->resident(matrices);
  const std::vector<std::uint32_t> zero_status(kTail, 0u);
  if (!factor_job ||
      !CheckSuccess(*factor_job, backend, "factor-tail", evidence.factor,
                    [&](const auto &output) {
                      return std::get<0>(output) == matrices &&
                             std::get<1>(output) == zero_status &&
                             std::get<2>(output) == zero_status;
                    }) ||
      !solve_job ||
      !CheckSuccess(*solve_job, backend, "solve-tail", evidence.solve,
                    [&](const auto &output) {
                      return std::get<0>(output).size() == kTail &&
                             std::get<1>(output) == zero_status;
                    }) ||
      !spectrum_job ||
      !CheckSuccess(*spectrum_job, backend, "spectrum-tail", evidence.spectrum,
                    [&](const auto &output) {
                      return std::get<0>(output) == matrices &&
                             std::get<1>(output) == zero_status;
                    })) {
    return false;
  }
  return true;
}

template <class T> [[nodiscard]] bool CheckFixedFamily() {
  FixedEvidence evidence{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckFixed<T>(backend, evidence)) {
      std::fprintf(stderr, "boundary modes fixed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

bool CheckFixedModes() {
  using F32 = rund::compute::Fixed<1, 31>;
  using F64 = rund::compute::Fixed<1, 63>;
  return CheckFixedFamily<F32>() && CheckFixedFamily<F64>();
}

} // namespace rund::node::test_contract::boundary_modes
