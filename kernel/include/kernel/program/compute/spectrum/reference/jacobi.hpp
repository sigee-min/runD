#pragma once

#include <kernel/program/compute/spectrum/reference/base.hpp>

#include <algorithm>

namespace rund::kernel::spectrum_reference_detail {

template <typename S>
[[nodiscard]] inline JacobiResult JacobiSymmetric(
    S *const a, S *const v, S *const values, const u64 n,
    const u32 max_iterations, const ComputeFixedFormat format) {
  if (a == nullptr || v == nullptr || values == nullptr) {
    return {};
  }
  std::fill_n(v, static_cast<std::size_t>(n * n), S{0});
  for (u64 i = 0u; i < n; ++i) {
    v[static_cast<std::size_t>(i * n + i)] = One<S>(format);
  }
  bool converged = false;
  for (u32 iteration = 0u; iteration < max_iterations; ++iteration) {
    u64 p = 0u;
    u64 q = 1u;
    S best = 0;
    for (u64 row = 0u; row < n; ++row) {
      for (u64 col = row + 1u; col < n; ++col) {
        const S candidate = a[static_cast<std::size_t>(row * n + col)];
        if (Magnitude(candidate) > Magnitude(best)) {
          best = candidate;
          p = row;
          q = col;
        }
      }
    }
    if (Magnitude(best) <= Magnitude(Epsilon<S>(format))) {
      converged = true;
      break;
    }
    const S app = a[static_cast<std::size_t>(p * n + p)];
    const S aqq = a[static_cast<std::size_t>(q * n + q)];
    const S apq = a[static_cast<std::size_t>(p * n + q)];
    const S tau = factor_reference_detail::DivFixed(
        factor_reference_detail::SubSat(aqq, app),
        factor_reference_detail::AddSat(apq, apq), format);
    const S tau_scaled = factor_reference_detail::MulFixed(
        tau, Quarter<S>(format), format);
    const S root_scaled = factor_reference_detail::SqrtFixed(
        factor_reference_detail::AddSat(
            Sixteenth<S>(format),
            factor_reference_detail::MulFixed(tau_scaled, tau_scaled, format)),
        format);
    const S sign_scaled = tau < 0 ? static_cast<S>(-Quarter<S>(format))
                                  : Quarter<S>(format);
    const S t = factor_reference_detail::DivFixed(
        sign_scaled,
        factor_reference_detail::AddSat(
            factor_reference_detail::MulFixed(AbsSat(tau), Quarter<S>(format),
                                              format),
            root_scaled),
        format);
    const S t_scaled = factor_reference_detail::MulFixed(
        t, Quarter<S>(format), format);
    const S c = factor_reference_detail::DivFixed(
        Quarter<S>(format),
        factor_reference_detail::SqrtFixed(
            factor_reference_detail::AddSat(
                Sixteenth<S>(format),
                factor_reference_detail::MulFixed(t_scaled, t_scaled, format)),
            format),
        format);
    const S s = factor_reference_detail::MulFixed(t, c, format);
    for (u64 k = 0u; k < n; ++k) {
      const S akp = a[static_cast<std::size_t>(k * n + p)];
      const S akq = a[static_cast<std::size_t>(k * n + q)];
      a[static_cast<std::size_t>(k * n + p)] = factor_reference_detail::SubSat(
          factor_reference_detail::MulFixed(c, akp, format),
          factor_reference_detail::MulFixed(s, akq, format));
      a[static_cast<std::size_t>(k * n + q)] = factor_reference_detail::AddSat(
          factor_reference_detail::MulFixed(s, akp, format),
          factor_reference_detail::MulFixed(c, akq, format));
    }
    for (u64 k = 0u; k < n; ++k) {
      const S apk = a[static_cast<std::size_t>(p * n + k)];
      const S aqk = a[static_cast<std::size_t>(q * n + k)];
      a[static_cast<std::size_t>(p * n + k)] = factor_reference_detail::SubSat(
          factor_reference_detail::MulFixed(c, apk, format),
          factor_reference_detail::MulFixed(s, aqk, format));
      a[static_cast<std::size_t>(q * n + k)] = factor_reference_detail::AddSat(
          factor_reference_detail::MulFixed(s, apk, format),
          factor_reference_detail::MulFixed(c, aqk, format));
    }
    for (u64 k = 0u; k < n; ++k) {
      const S vkp = v[static_cast<std::size_t>(k * n + p)];
      const S vkq = v[static_cast<std::size_t>(k * n + q)];
      v[static_cast<std::size_t>(k * n + p)] = factor_reference_detail::SubSat(
          factor_reference_detail::MulFixed(c, vkp, format),
          factor_reference_detail::MulFixed(s, vkq, format));
      v[static_cast<std::size_t>(k * n + q)] = factor_reference_detail::AddSat(
          factor_reference_detail::MulFixed(s, vkp, format),
          factor_reference_detail::MulFixed(c, vkq, format));
    }
  }
  for (u64 i = 0u; i < n; ++i) {
    values[static_cast<std::size_t>(i)] =
        a[static_cast<std::size_t>(i * n + i)];
  }
  return JacobiResult{.converged = converged};
}

} // namespace rund::kernel::spectrum_reference_detail
