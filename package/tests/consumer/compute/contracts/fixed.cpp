#include "common.hpp"

#include <type_traits>

static_assert(sizeof(rund::compute::Fixed<16, 16>) == sizeof(std::int32_t));
static_assert(sizeof(rund::compute::Fixed<20, 44>) == sizeof(std::int64_t));
static_assert(
    std::is_same_v<typename rund::compute::Fixed<16, 16>::Raw, std::int32_t>);
static_assert(
    std::is_same_v<typename rund::compute::Fixed<20, 44>::Raw, std::int64_t>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<16, 16>, float>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<16, 16>, double>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<20, 44>, float>);
static_assert(!std::is_convertible_v<rund::compute::Fixed<20, 44>, double>);
static_assert(!CastsToFloat<rund::compute::Fixed<16, 16>>);
static_assert(!CastsToDouble<rund::compute::Fixed<16, 16>>);
static_assert(!CastsToFloat<rund::compute::Fixed<20, 44>>);
static_assert(!CastsToDouble<rund::compute::Fixed<20, 44>>);
static_assert(!std::constructible_from<rund::compute::Fixed<16, 16>, float>);
static_assert(!std::constructible_from<rund::compute::Fixed<32, 32>, double>);
static_assert(!DefinesFixed<std::numeric_limits<unsigned>::max(), 33u>);
static_assert(!AddsFloatExpression<rund::compute::Fixed<16, 16>>);
using Fixed16x16 = rund::compute::Fixed<16, 16>;
using Fixed1x31 = rund::compute::Fixed<1, 31>;
using Fixed8x24 = rund::compute::Fixed<8, 24>;
static_assert(!AddsExpressions<Fixed16x16, Fixed1x31>);
static_assert(!SubtractsExpressions<Fixed16x16, Fixed8x24>);
static_assert(!MultipliesExpressions<Fixed16x16, Fixed8x24>);
static_assert(!DividesExpressions<Fixed16x16, Fixed8x24>);
static_assert(!MinimizesExpressions<Fixed16x16, Fixed8x24>);
static_assert(!MaximizesExpressions<Fixed16x16, Fixed8x24>);
