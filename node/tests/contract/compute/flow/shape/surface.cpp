#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {
template <class T>
concept CanLu = requires(T value) { std::move(value).lu(); };
template <class T>
concept CanQr = requires(T value) { std::move(value).qr(); };
template <class T>
concept CanCholesky = requires(T value) { std::move(value).cholesky(); };
template <class T>
concept CanEigen = requires(T value) { std::move(value).eigen(); };
template <class T>
concept CanSvd = requires(T value) { std::move(value).svd(); };
template <class T>
concept CanProjectVectors = requires(T value) { std::move(value).vectors(); };
template <class T>
concept CanMap = requires(T value) {
  std::move(value).map("typed-map", [](auto item) { return item; });
};
template <class T>
concept CanFilter = requires(T value) {
  std::move(value).filter([](auto item) { return item == item; });
};
template <class T>
concept CanScan = requires(T value) {
  std::move(value).scan(rund::compute::Scan::InclusiveSum);
};
template <class T>
concept CanReduce = requires(T value) { std::move(value).reduce(); };
template <class T>
concept CanSort = requires(T value) { std::move(value).sort(); };
template <class T>
concept CanCount = requires(T value) { std::move(value).count(); };
template <class T>
concept CanWindow =
    requires(T value) { std::move(value).window(rund::compute::WindowSpec{}); };
template <class T>
concept CanStencil = requires(T value) {
  std::move(value).stencil(rund::compute::WindowSpec{});
};
template <class T>
concept CanMatrix =
    requires(T value) { std::move(value).template matrix<2u, 2u>(); };
template <class T>
concept CanZeroMatrix =
    requires(T value) { std::move(value).template matrix<0u, 2u>(); };
template <class T>
concept CanOverflowMatrix = requires(T value) {
  std::move(value)
      .template matrix<std::numeric_limits<std::size_t>::max(), 2u>();
};
template <class T>
concept CanComplex = requires(T value) { std::move(value).complex(); };
template <class T>
concept CanFourier = requires(T value) { std::move(value).fourier(); };
template <class T>
concept CanPacked = requires(T value) { std::move(value).packed(); };
template <class T>
concept CanValues = requires(T value) { std::move(value).values(); };
template <class T>
concept CanStatus = requires(T value) { std::move(value).status(); };
template <class T, class Range>
concept CanDynamicMatrixSolve = requires(T value, Range &rhs) {
  std::move(value).solve(rhs, rund::compute::FactorOp::Lu, 1u);
};
template <class T, class Range>
concept CanStaticMatrixSolve = requires(T value, Range &rhs) {
  std::move(value).template solve<rund::compute::FactorOp::Lu, 1u>(rhs);
};

using namespace rund::compute;
using StaticIntegerMatrix =
    decltype(on(Target::cpu(),
                std::declval<const std::array<std::int32_t, 4> &>())
                 .matrix<2u, 2u>());
using DynamicIntegerMatrix =
    decltype(on(Target::cpu(),
                std::declval<const std::array<std::int32_t, 4> &>())
                 .matrix({2u, 2u, 1u}));
static_assert(std::same_as<DynamicIntegerMatrix,
                           Flow<std::int32_t(std::int32_t), stage::Matrix<>>>);
static_assert(
    std::same_as<StaticIntegerMatrix,
                 Flow<std::int32_t(std::int32_t), stage::Matrix<2u, 2u, 1u>>>);
using StaticIntegerTranspose =
    decltype(std::declval<StaticIntegerMatrix>().transpose());
static_assert(
    std::same_as<StaticIntegerTranspose,
                 Flow<std::int32_t(std::int32_t), stage::Matrix<2u, 2u, 1u>>>);
using StaticIntegerProduct =
    decltype(std::declval<StaticIntegerMatrix>().template matmul<2u, 3u>(
        std::declval<const std::array<std::int32_t, 6> &>()));
static_assert(std::same_as<StaticIntegerProduct,
                           Flow<std::int32_t(std::int32_t, std::int32_t),
                                stage::Matrix<2u, 3u, 1u>>>);
template <class Stage, class Range>
concept CanMismatchedMatmul = requires(Stage stage, Range &range) {
  std::move(stage).template matmul<3u, 2u>(range);
};
static_assert(!CanMismatchedMatmul<StaticIntegerMatrix,
                                   const std::array<std::int32_t, 6>>);
using StaticIntegerRhs = std::array<std::int32_t, 2u>;
static_assert(!CanLu<StaticIntegerMatrix>);
static_assert(!CanQr<StaticIntegerMatrix>);
static_assert(!CanCholesky<StaticIntegerMatrix>);
static_assert(!CanEigen<StaticIntegerMatrix>);
static_assert(!CanMap<StaticIntegerMatrix>);
static_assert(!CanFilter<StaticIntegerMatrix>);
static_assert(!CanScan<StaticIntegerMatrix>);
static_assert(!CanReduce<StaticIntegerMatrix>);
static_assert(!CanSort<StaticIntegerMatrix>);
static_assert(!CanCount<StaticIntegerMatrix>);
static_assert(!CanMatrix<StaticIntegerMatrix>);
static_assert(!CanComplex<StaticIntegerMatrix>);
static_assert(
    !CanDynamicMatrixSolve<StaticIntegerMatrix, const StaticIntegerRhs> &&
    !CanStaticMatrixSolve<StaticIntegerMatrix, const StaticIntegerRhs>);

using StaticFixedMatrix =
    decltype(on(Target::cpu(),
                std::declval<const std::array<Fixed<1, 31>, 4> &>())
                 .matrix<2u, 2u>());
using StaticLu = decltype(std::declval<StaticFixedMatrix>().lu());
static_assert(
    std::same_as<StaticLu, Flow<Fixed<1, 31>(Fixed<1, 31>),
                                stage::Factor<FactorOp::Lu, 2u, 2u, 1u>>>);
using StaticLuPacked = decltype(std::declval<StaticLu>().packed());
static_assert(
    std::same_as<StaticLuPacked,
                 Flow<Fixed<1, 31>(Fixed<1, 31>), stage::Matrix<2u, 2u, 1u>>>);
using StaticSolve = decltype(std::declval<StaticLu>().template solve<1u>(
    std::declval<const std::array<Fixed<1, 31>, 2> &>()));
static_assert(
    std::same_as<StaticSolve, Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                                   stage::Solve<2u, 1u, 1u>>>);
using StaticSolveValues = decltype(std::declval<StaticSolve>().values());
static_assert(std::same_as<StaticSolveValues,
                           Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                                stage::Matrix<2u, 1u, 1u>>>);
using StaticRhs = std::array<Fixed<1, 31>, 2u>;
using StaticMatrixSolve =
    decltype(std::declval<StaticFixedMatrix>().template solve<FactorOp::Lu, 1u>(
        std::declval<const StaticRhs &>()));
static_assert(std::same_as<StaticMatrixSolve,
                           Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                                stage::Solve<2u, 1u, 1u>>>);
using DynamicMatrixSolve = decltype(std::declval<StaticFixedMatrix>().solve(
    std::declval<const StaticRhs &>(), FactorOp::Qr, 1u));
static_assert(std::same_as<DynamicMatrixSolve,
                           Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                                stage::Solve<2u, stage::Dynamic, 1u>>>);
using DeferredStaticMatrix =
    decltype(on(Target::cpu(2u))
                 .map<Fixed<1, 31>>(
                     "deferred-matrix-solve", 4u,
                     [](auto value) { return quantize<Fixed<1, 31>>(value); })
                 .matrix<2u, 2u>());
using DeferredMatrixSolve =
    decltype(std::declval<DeferredStaticMatrix>()
                 .template solve<FactorOp::Cholesky, 1u>());
static_assert(std::same_as<DeferredMatrixSolve,
                           Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                                stage::Solve<2u, 1u, 1u>, input::Deferred>>);
using DeferredDynamicMatrixSolve =
    decltype(std::declval<DeferredStaticMatrix>().solve(FactorOp::Qr, 1u));
static_assert(
    std::same_as<DeferredDynamicMatrixSolve,
                 Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                      stage::Solve<2u, stage::Dynamic, 1u>, input::Deferred>>);
using MatrixStageSolve =
    decltype(std::declval<StaticFixedMatrix>().pipe([](auto matrix) {
      return matrix.template solve<FactorOp::Lu, 2u>(matrix);
    }));
static_assert(
    std::same_as<MatrixStageSolve,
                 Flow<Fixed<1, 31>(Fixed<1, 31>), stage::Solve<2u, 2u, 1u>>>);
using DynamicMatrixStageSolve = decltype(std::declval<StaticFixedMatrix>().pipe(
    [](auto matrix) { return matrix.solve(matrix, FactorOp::Qr, 2u); }));
static_assert(std::same_as<DynamicMatrixStageSolve,
                           Flow<Fixed<1, 31>(Fixed<1, 31>),
                                stage::Solve<2u, stage::Dynamic, 1u>>>);
using StaticMatrixRef = StageRef<Fixed<1, 31>, stage::Matrix<2u, 2u, 1u>>;
static_assert(!CanMap<StaticMatrixRef>);
static_assert(CanDynamicMatrixSolve<StaticFixedMatrix, const StaticRhs> &&
              CanStaticMatrixSolve<StaticFixedMatrix, const StaticRhs>);

using RectFixedMatrix =
    decltype(on(Target::cpu(),
                std::declval<const std::array<Fixed<1, 31>, 6> &>())
                 .matrix<3u, 2u>());
static_assert(!CanLu<RectFixedMatrix>);
static_assert(CanQr<RectFixedMatrix>);
static_assert(!CanCholesky<RectFixedMatrix>);
static_assert(!CanEigen<RectFixedMatrix>);
using StaticSvd = decltype(std::declval<RectFixedMatrix>()
                               .template svd<SpectrumVectors::Thin>());
static_assert(
    std::same_as<StaticSvd,
                 Flow<Fixed<1, 31>(Fixed<1, 31>),
                      stage::Spectrum<SpectrumOp::Svd, SpectrumVectors::Thin,
                                      3u, 2u, 1u>>>);
using StaticSvdVectors = decltype(std::declval<StaticSvd>().vectors());
static_assert(
    std::same_as<StaticSvdVectors,
                 Flow<Fixed<1, 31>(Fixed<1, 31>), stage::Matrix<3u, 2u, 1u>>>);
using StaticSvdValues = decltype(std::declval<StaticFixedMatrix>()
                                     .template svd<SpectrumVectors::Values>());
static_assert(!CanProjectVectors<StaticSvdValues>);
static_assert(!CanDynamicMatrixSolve<RectFixedMatrix, const StaticRhs> &&
              !CanStaticMatrixSolve<RectFixedMatrix, const StaticRhs>);

using Sequence = decltype(on(
    Target::cpu(), std::declval<const std::array<std::int32_t, 4> &>()));
using Bounded = decltype(std::declval<Sequence>().filter(
    [](auto item) { return item > 0; }));
using Scalar = decltype(std::declval<Sequence>().reduce());
using FixedSequence = decltype(on(
    Target::cpu(), std::declval<const std::array<Fixed<1, 31>, 4> &>()));
using Complex =
    decltype(on(Target::cpu(),
                std::declval<const std::array<Fixed<1, 31>, 4> &>())
                 .complex(std::declval<const std::array<Fixed<1, 31>, 4> &>()));
static_assert(
    std::same_as<Complex, Flow<Fixed<1, 31>(Fixed<1, 31>, Fixed<1, 31>),
                               stage::Complex>>);
using ComplexCollect = decltype(std::declval<Complex>().collect());
static_assert(std::same_as<ComplexCollect,
                           Result<std::tuple<std::vector<Fixed<1, 31>>,
                                             std::vector<Fixed<1, 31>>>>>);
using DynamicLu =
    decltype(on(Target::cpu(),
                std::declval<const std::array<Fixed<1, 31>, 4> &>())
                 .matrix({2u, 2u, 1u})
                 .lu());
static_assert(std::same_as<DynamicLu, Flow<Fixed<1, 31>(Fixed<1, 31>),
                                           stage::Factor<FactorOp::Lu>>>);
using DynamicSpectrum =
    decltype(on(Target::cpu(),
                std::declval<const std::array<Fixed<1, 31>, 4> &>())
                 .matrix({2u, 2u, 1u})
                 .svd<SpectrumVectors::Thin>());
static_assert(std::same_as<
              DynamicSpectrum,
              Flow<Fixed<1, 31>(Fixed<1, 31>),
                   stage::Spectrum<SpectrumOp::Svd, SpectrumVectors::Thin>>>);
static_assert(CanMap<Sequence> && CanFilter<Sequence> && CanScan<Sequence> &&
              CanReduce<Sequence> && CanSort<Sequence> && CanCount<Sequence> &&
              CanMatrix<Sequence> && CanWindow<Sequence> &&
              !CanStencil<Sequence>);
static_assert(!CanZeroMatrix<Sequence> && !CanOverflowMatrix<Sequence>);
static_assert(!CanLu<FixedSequence> && !CanQr<FixedSequence> &&
              !CanCholesky<FixedSequence> && !CanSvd<FixedSequence> &&
              !CanEigen<FixedSequence>);
static_assert(CanMap<Bounded> && CanFilter<Bounded> && CanScan<Bounded> &&
              CanReduce<Bounded> && CanSort<Bounded> && CanCount<Bounded>);
static_assert(!CanMatrix<Bounded> && !CanComplex<Bounded>);
static_assert(CanMap<Scalar> && !CanFilter<Scalar> && !CanScan<Scalar> &&
              !CanReduce<Scalar> && !CanSort<Scalar> && !CanCount<Scalar> &&
              !CanMatrix<Scalar>);
static_assert(CanFourier<Complex> && !CanMap<Complex> && !CanFilter<Complex> &&
              !CanScan<Complex> && !CanReduce<Complex> && !CanSort<Complex> &&
              !CanMatrix<Complex>);
static_assert(CanPacked<StaticLu> && CanStatus<StaticLu> && !CanMap<StaticLu> &&
              !CanScan<StaticLu> && !CanProjectVectors<StaticLu>);
static_assert(CanValues<StaticSolve> && CanStatus<StaticSolve> &&
              !CanMap<StaticSolve> && !CanScan<StaticSolve> &&
              !CanPacked<StaticSolve>);
static_assert(CanValues<StaticSvd> && CanProjectVectors<StaticSvd> &&
              CanStatus<StaticSvd> && !CanMap<StaticSvd> &&
              !CanScan<StaticSvd> && !CanPacked<StaticSvd>);

using OrderedBranchProgram =
    decltype(on(Target::cpu())
                 .map<std::uint32_t>("shape-branch", 5u,
                                     [](auto value) { return value; })
                 .branch([](auto values) {
                   const auto doubled =
                       values.map("shape-branch-double",
                                  [](auto value) { return value * 2u; });
                   return outputs(values, doubled, values.sort(),
                                  values.reduce(Reduce::Sum));
                 })
                 .compile());
static_assert(
    std::same_as<OrderedBranchProgram,
                 Result<Program<Outputs<
                     std::uint32_t, std::uint32_t, std::uint32_t,
                     rund::compute::Scalar<std::uint32_t>>(std::uint32_t)>>>);

using BoundedMapProgram =
    decltype(on(Target::cpu(2u))
                 .map<std::int64_t>("shape-bounded-map", 4u,
                                    [](auto value) { return value; })
                 .filter([](auto value) { return value > 2; })
                 .map("shape-bounded-double",
                      [](auto value) { return value * 2; })
                 .compile());
static_assert(
    std::same_as<
        BoundedMapProgram,
        Result<Program<rund::compute::Bounded<std::int64_t>(std::int64_t)>>>);

using BoundedCountProgram =
    decltype(on(Target::cpu(2u))
                 .map<std::int64_t>("shape-bounded-count", 4u,
                                    [](auto value) { return value; })
                 .filter([](auto value) { return value > 2; })
                 .count()
                 .compile());
static_assert(std::same_as<BoundedCountProgram,
                           Result<Program<std::uint64_t(std::int64_t)>>>);

using SplitTransformProgram =
    decltype(on(Target::cpu(2u))
                 .map<Fixed<1, 31>>(
                     "shape-transform", 4u,
                     [](auto value) { return quantize<Fixed<1, 31>>(value); })
                 .complex()
                 .fourier()
                 .compile());
static_assert(std::same_as<SplitTransformProgram,
                           Result<Program<Outputs<Fixed<1, 31>, Fixed<1, 31>>(
                               Fixed<1, 31>, Fixed<1, 31>)>>>);

using FactorProgram =
    decltype(on(Target::cpu(2u))
                 .map<Fixed<1, 31>>(
                     "shape-factor", 4u,
                     [](auto value) { return quantize<Fixed<1, 31>>(value); })
                 .matrix({2u, 2u, 1u})
                 .lu()
                 .compile());
static_assert(
    std::same_as<FactorProgram,
                 Result<Program<Outputs<Fixed<1, 31>, std::uint32_t,
                                        std::uint32_t>(Fixed<1, 31>)>>>);

using SolveProgram =
    decltype(on(Target::cpu(2u))
                 .map<Fixed<1, 31>>(
                     "shape-solve", 4u,
                     [](auto value) { return quantize<Fixed<1, 31>>(value); })
                 .matrix({2u, 2u, 1u})
                 .lu()
                 .solve(2u)
                 .compile());
static_assert(std::same_as<SolveProgram,
                           Result<Program<Outputs<Fixed<1, 31>, std::uint32_t>(
                               Fixed<1, 31>, Fixed<1, 31>)>>>);

using SpectrumProgram =
    decltype(on(Target::cpu(2u))
                 .map<Fixed<1, 31>>(
                     "shape-spectrum", 4u,
                     [](auto value) { return quantize<Fixed<1, 31>>(value); })
                 .matrix({2u, 2u, 1u})
                 .svd<SpectrumVectors::Thin>()
                 .compile());
static_assert(
    std::same_as<SpectrumProgram,
                 Result<Program<Outputs<Fixed<1, 31>, Fixed<1, 31>,
                                        std::uint32_t>(Fixed<1, 31>)>>>);

} // namespace
