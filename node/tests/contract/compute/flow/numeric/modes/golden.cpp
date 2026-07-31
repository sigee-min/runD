#include "../local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <tuple>
#include <vector>

namespace rund::node::test_contract::numeric {
namespace {

template <class T>
[[nodiscard]] int CheckAbsoluteGolden(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  const auto value = [](const std::int64_t integer) {
    return T::from_raw(static_cast<Raw>(integer) << T::fraction_bits);
  };
  const T zero = T::zero();

  const std::array<T, 4u> left{value(1), zero, zero, value(2)};
  const std::array<T, 4u> right{value(3), zero, zero, value(4)};
  auto product = on(rund::node::test_contract::target_for(backend), left)
                     .matrix({2u, 2u, 1u})
                     .matmul(right, {2u, 2u, 1u})
                     .collect();
  if (!product || *product != std::vector<T>{value(3), zero, zero, value(8)}) {
    return 1;
  }

  const std::array<T, 4u> impulse{value(1), zero, zero, zero};
  const std::array<T, 4u> imag{zero, zero, zero, zero};
  auto transformed = on(rund::node::test_contract::target_for(backend), impulse)
                         .complex(imag)
                         .fourier()
                         .real()
                         .collect();
  if (!transformed ||
      *transformed != std::vector<T>{value(1), value(1), value(1), value(1)}) {
    return 2;
  }

  const std::array<T, 4u> spd{value(4), zero, zero, value(9)};
  auto cholesky = on(rund::node::test_contract::target_for(backend), spd)
                      .template matrix<2u, 2u>()
                      .cholesky()
                      .packed()
                      .collect();
  if (!cholesky ||
      *cholesky != std::vector<T>{value(2), zero, zero, value(3)}) {
    return 3;
  }

  const std::array<T, 4u> diagonal{value(4), zero, zero, value(9)};
  const std::array<T, 2u> rhs{value(8), value(18)};
  auto solved = on(rund::node::test_contract::target_for(backend), diagonal)
                    .template matrix<2u, 2u>()
                    .cholesky()
                    .template solve<1u>(rhs)
                    .values()
                    .collect();
  if (!solved || *solved != std::vector<T>{value(2), value(2)}) {
    return 4;
  }

  const std::array<T, 4u> spectrum{value(2), zero, zero, value(3)};
  auto singular = on(rund::node::test_contract::target_for(backend), spectrum)
                      .template matrix<2u, 2u>()
                      .template svd<SpectrumVectors::Values>(128u)
                      .values()
                      .collect();
  return singular && *singular == std::vector<T>{value(3), value(2)} ? 0 : 5;
}

template <class T, rund::compute::Rounding Round,
          rund::compute::Overflow OverflowMode>
[[nodiscard]] rund::compute::Result<std::vector<T>>
MatrixScalar(const rund::compute::Backend backend, const T lhs, const T rhs) {
  using namespace rund::compute;
  const std::array<T, 1u> left{lhs};
  const std::array<T, 1u> right{rhs};
  auto program =
      on(rund::node::test_contract::target_for(backend))
          .template map<T>("matrix-policy-scalar", 1u,
                           [](auto value) {
                             return quantize<T, Round, OverflowMode>(value);
                           })
          .template matrix<1u, 1u>()
          .template matmul<1u, 1u>()
          .compile();
  if (!program) {
    return Result<std::vector<T>>::fail(program.reason());
  }
  auto job = program->resident(left, right);
  if (!job) {
    return Result<std::vector<T>>::fail(job.reason());
  }
  const Status ran = job->run();
  if (!ran) {
    return Result<std::vector<T>>::fail(ran.reason());
  }
  return job->read();
}

template <class T>
[[nodiscard]] int
CheckMatrixPolicyGolden(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  const T raw_one = T::from_raw(Raw{1});
  const T negative_raw_one = T::from_raw(Raw{-1});
  const T half =
      T::from_raw(static_cast<Raw>(Raw{1} << (T::fraction_bits - 1u)));

  const auto nearest =
      MatrixScalar<T, Rounding::NearestEven, Overflow::Saturate>(backend,
                                                                 raw_one, half);
  const auto upward =
      MatrixScalar<T, Rounding::Up, Overflow::Saturate>(backend, raw_one, half);
  const auto downward = MatrixScalar<T, Rounding::Down, Overflow::Saturate>(
      backend, negative_raw_one, half);
  if (!nearest || nearest->size() != 1u || (*nearest)[0].raw() != Raw{0} ||
      !upward || upward->size() != 1u || (*upward)[0].raw() != Raw{1} ||
      !downward || downward->size() != 1u || (*downward)[0].raw() != Raw{-1}) {
    const auto raw = [](const auto &result) -> long long {
      return result && result->size() == 1u
                 ? static_cast<long long>((*result)[0].raw())
                 : std::numeric_limits<long long>::min();
    };
    const auto reason = [](const auto &result) {
      return result ? std::string_view{"ok"} : result.error();
    };
    const auto nearest_reason = reason(nearest);
    const auto upward_reason = reason(upward);
    const auto downward_reason = reason(downward);
    std::fprintf(stderr,
                 "matrix policy rounding backend=%u nearest=%lld upward=%lld "
                 "downward=%lld reasons=%.*s/%.*s/%.*s\n",
                 static_cast<unsigned>(backend), raw(nearest), raw(upward),
                 raw(downward), static_cast<int>(nearest_reason.size()),
                 nearest_reason.data(), static_cast<int>(upward_reason.size()),
                 upward_reason.data(), static_cast<int>(downward_reason.size()),
                 downward_reason.data());
    return 1;
  }

  const T two = T::from_raw(static_cast<Raw>(Raw{2} << T::fraction_bits));
  const auto saturated =
      MatrixScalar<T, Rounding::TowardZero, Overflow::Saturate>(backend,
                                                                T::max(), two);
  const auto wrapped = MatrixScalar<T, Rounding::TowardZero, Overflow::Wrap>(
      backend, T::max(), two);
  if (!saturated || saturated->size() != 1u || (*saturated)[0] != T::max() ||
      !wrapped || wrapped->size() != 1u || (*wrapped)[0].raw() != Raw{-2}) {
    return 2;
  }
  return 0;
}

template <class T, rund::compute::Overflow OverflowMode>
[[nodiscard]] rund::compute::Result<std::vector<T>>
SolveOverflowScalar(const rund::compute::Backend backend, const T diagonal,
                    const T rhs) {
  using namespace rund::compute;
  auto program =
      on(rund::node::test_contract::target_for(backend))
          .template map<T>(
              "solve-overflow-scalar", 1u,
              [](auto value) {
                return quantize<T, Rounding::TowardZero, OverflowMode,
                                Approximation::Deterministic>(value);
              })
          .template matrix<1u, 1u>()
          .lu()
          .template solve<1u>()
          .compile();
  if (!program) {
    return Result<std::vector<T>>::fail(program.reason());
  }
  const std::array<T, 1u> matrix{diagonal};
  const std::array<T, 1u> right_hand_side{rhs};
  auto job = program->resident(matrix, right_hand_side);
  if (!job) {
    return Result<std::vector<T>>::fail(job.reason());
  }
  const Status ran = job->run();
  if (!ran) {
    return Result<std::vector<T>>::fail(ran.reason());
  }
  return job->template read<0u>();
}

template <class T>
[[nodiscard]] int
CheckSolveOverflowGolden(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  const T half =
      T::from_raw(static_cast<Raw>(Raw{1} << (T::fraction_bits - 1u)));
  const auto saturated =
      SolveOverflowScalar<T, Overflow::Saturate>(backend, half, T::max());
  const auto wrapped =
      SolveOverflowScalar<T, Overflow::Wrap>(backend, half, T::max());
  if (!saturated || saturated->size() != 1u || (*saturated)[0] != T::max() ||
      !wrapped || wrapped->size() != 1u || (*wrapped)[0].raw() != Raw{-2}) {
    return 1;
  }
  return 0;
}

} // namespace

int CheckGolden32(const rund::compute::Backend backend) {
  return CheckAbsoluteGolden<rund::compute::Fixed<16, 16>>(backend);
}

int CheckGolden64(const rund::compute::Backend backend) {
  return CheckAbsoluteGolden<rund::compute::Fixed<20, 44>>(backend);
}

int CheckMatrix32(const rund::compute::Backend backend) {
  return CheckMatrixPolicyGolden<rund::compute::Fixed<16, 16>>(backend);
}

int CheckMatrix64(const rund::compute::Backend backend) {
  return CheckMatrixPolicyGolden<rund::compute::Fixed<20, 44>>(backend);
}

int CheckSolve32(const rund::compute::Backend backend) {
  return CheckSolveOverflowGolden<rund::compute::Fixed<16, 16>>(backend);
}

int CheckSolve64(const rund::compute::Backend backend) {
  return CheckSolveOverflowGolden<rund::compute::Fixed<20, 44>>(backend);
}

} // namespace rund::node::test_contract::numeric
