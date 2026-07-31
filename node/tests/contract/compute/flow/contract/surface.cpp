#include "model.hpp"

#include <concepts>
#include <initializer_list>
#include <memory_resource>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund_node_flow_contract {

template <class T>
concept AcceptsRvalueFlowInput = requires(T value) {
  rund::compute::on(rund::compute::Target::cpu(), std::move(value));
};

template <class Range>
concept AcceptsFlowRange = requires(Range &value) {
  rund::compute::on(rund::compute::Target::cpu(), value);
};

template <class Range>
concept AcceptsConstFlowRange = requires(const Range &value) {
  rund::compute::on(rund::compute::Target::cpu(), value);
};

template <class T>
concept HasWorkers = requires(T value) {
  { value.workers() } -> std::same_as<std::uint32_t>;
};

template <class T>
concept CanSort = requires(T value) { std::move(value).sort(); };

template <class A, class B>
concept CanZip = requires(const A &left, const B &right) {
  rund::compute::zip(left, right);
};
template <class T>
concept CanSin = requires(const T &value) { rund::compute::sin(value); };
template <class T>
concept CanFixedUnaryFunctions = requires(const T &value) {
  rund::compute::neg_positive_fixed(value);
  rund::compute::recip(value);
  rund::compute::sqrt(value);
  rund::compute::rsqrt(value);
  rund::compute::sin(value);
  rund::compute::cos(value);
  rund::compute::tan(value);
  rund::compute::exp(value);
  rund::compute::log(value);
};
template <class T>
concept CanFixedBinaryFunctions = requires(const T &value) {
  rund::compute::mul_fixed(value, value);
  rund::compute::mul_fixed_scaled(value, value);
  rund::compute::mul_unsigned_fixed(value, value);
  rund::compute::atan2(value, value);
};
template <class T>
concept CanDivFixed =
    requires(const T &value) { rund::compute::div_fixed(value, value); };
template <class T>
concept CanFixedTernaryFunctions = requires(const T &value) {
  rund::compute::mul_add_fixed(value, value, value);
};
template <class T>
concept CanQuantizeToFixed = requires(const T &value) {
  rund::compute::quantize<rund::compute::Fixed<1, 31>>(value);
};
template <class T>
concept CanShift = requires(const T &value) { rund::compute::shl<1u>(value); };
template <class T>
concept CanArithmeticShift =
    requires(const T &value) { rund::compute::shr_arithmetic<1u>(value); };
template <class T>
concept CanUnsignedStorageSaturate =
    requires(const T &value) { rund::compute::add_sat_unsigned(value, value); };

template <class T>
concept AcceptsRvalueGather =
    requires(rund::compute::Flow<std::int32_t(std::int32_t)> flow, T value) {
      std::move(flow).gather(std::move(value));
    };

template <class Range>
concept AcceptsGatherRange =
    requires(rund::compute::Flow<std::int32_t(std::int32_t)> flow,
             Range &value) { std::move(flow).gather(value); };

static_assert(!AcceptsRvalueFlowInput<std::vector<std::int32_t>>);
static_assert(!AcceptsRvalueFlowInput<std::array<std::int32_t, 4>>);
static_assert(!AcceptsRvalueFlowInput<std::span<std::int32_t>>);
static_assert(AcceptsFlowRange<std::int32_t[4]>);
static_assert(AcceptsConstFlowRange<std::int32_t[4]>);
static_assert(AcceptsFlowRange<ContiguousInput<std::int32_t, 4>>);
static_assert(AcceptsConstFlowRange<ContiguousInput<std::int32_t, 4>>);
static_assert(AcceptsFlowRange<std::pmr::vector<std::int32_t>>);
static_assert(AcceptsConstFlowRange<std::pmr::vector<std::int32_t>>);
static_assert(!AcceptsFlowRange<std::initializer_list<std::int32_t>>);
static_assert(!AcceptsFlowRange<std::vector<bool>>);
static_assert(!AcceptsFlowRange<std::array<std::int16_t, 4>>);
static_assert(AcceptsFlowRange<std::vector<std::int32_t>>);
static_assert(AcceptsFlowRange<std::vector<rund::compute::Fixed<1, 63>>>);
static_assert(!AcceptsFlowRange<std::vector<float>>);
static_assert(!AcceptsFlowRange<std::vector<double>>);
static_assert(HasWorkers<rund::compute::Target>);
static_assert(!HasWorkers<rund::compute::Backend>);
using IntFlow = rund::compute::Flow<std::int32_t(std::int32_t)>;
using DeferredIntFlow =
    rund::compute::detail::DeferredFlow<std::int32_t(std::int32_t)>;
using ScalarFlow = decltype(std::declval<IntFlow>().reduce());
using DeferredScalarFlow = decltype(std::declval<DeferredIntFlow>().reduce());
using IntStage =
    rund::compute::StageRef<std::int32_t, rund::compute::stage::Exact>;
using UintStage =
    rund::compute::StageRef<std::uint32_t, rund::compute::stage::Exact>;
using HeterogeneousZip = decltype(rund::compute::zip(
    std::declval<const IntStage &>(), std::declval<const UintStage &>()));
using HeterogeneousRecord = decltype(rund::compute::record(
    std::declval<const IntStage &>(), std::declval<const UintStage &>()));
static_assert(
    std::same_as<decltype(rund::compute::on(rund::compute::Target::cpu())),
                 rund::compute::FlowBuilder>);
static_assert(std::same_as<
              decltype(rund::compute::on(rund::compute::Target::cpu(),
                                         std::declval<std::int32_t (&)[4]>())),
              IntFlow>);
static_assert(
    std::same_as<decltype(rund::compute::on(
                     rund::compute::Target::cpu(),
                     std::declval<const ContiguousInput<std::int32_t, 4> &>())),
                 IntFlow>);
static_assert(
    std::same_as<decltype(rund::compute::on(
                     rund::compute::Target::cpu(),
                     std::declval<std::pmr::vector<std::int32_t> &>())),
                 IntFlow>);
static_assert(
    std::same_as<decltype(rund::compute::on(
                     rund::compute::Target::cpu(),
                     std::declval<const std::pmr::vector<std::int32_t> &>())),
                 IntFlow>);
static_assert(std::same_as<ScalarFlow,
                           rund::compute::Flow<std::int32_t(std::int32_t),
                                               rund::compute::stage::Scalar>>);
static_assert(
    std::same_as<DeferredScalarFlow, rund::compute::detail::DeferredFlow<
                                         std::int32_t(std::int32_t),
                                         rund::compute::stage::Scalar>>);
static_assert(!CanSort<ScalarFlow>);
static_assert(!CanSort<DeferredScalarFlow>);
static_assert(CanSin<rund::compute::Expr<rund::compute::Fixed<1, 31>>>);
static_assert(!CanSin<rund::compute::Expr<std::int32_t>>);
using FixedExpr = rund::compute::Expr<rund::compute::Fixed<1, 31>>;
using IntegerExpr = rund::compute::Expr<std::int32_t>;
static_assert(CanFixedUnaryFunctions<FixedExpr>);
static_assert(!CanFixedUnaryFunctions<IntegerExpr>);
static_assert(CanFixedBinaryFunctions<FixedExpr>);
static_assert(!CanFixedBinaryFunctions<IntegerExpr>);
static_assert(CanDivFixed<FixedExpr>);
static_assert(!CanDivFixed<IntegerExpr>);
static_assert(CanFixedTernaryFunctions<FixedExpr>);
static_assert(!CanFixedTernaryFunctions<IntegerExpr>);
static_assert(CanQuantizeToFixed<FixedExpr>);
static_assert(!CanQuantizeToFixed<IntegerExpr>);
static_assert(CanShift<rund::compute::Expr<std::uint32_t>>);
static_assert(CanShift<rund::compute::Expr<rund::compute::Fixed<1, 31>>>);
static_assert(CanArithmeticShift<IntegerExpr>);
static_assert(!CanArithmeticShift<rund::compute::Expr<std::uint32_t>>);
static_assert(CanArithmeticShift<FixedExpr>);
static_assert(!CanUnsignedStorageSaturate<IntegerExpr>);
static_assert(CanUnsignedStorageSaturate<rund::compute::Expr<std::uint32_t>>);
static_assert(CanUnsignedStorageSaturate<FixedExpr>);
static_assert(!AcceptsRvalueGather<std::vector<std::uint32_t>>);
static_assert(!AcceptsRvalueGather<std::array<std::uint32_t, 4>>);
static_assert(!AcceptsRvalueGather<std::span<const std::uint32_t>>);
static_assert(AcceptsGatherRange<std::uint32_t[4]>);
static_assert(AcceptsGatherRange<const std::uint32_t[4]>);
static_assert(AcceptsGatherRange<ContiguousInput<std::uint32_t, 4>>);
static_assert(AcceptsGatherRange<const ContiguousInput<std::uint32_t, 4>>);
static_assert(!AcceptsGatherRange<std::int32_t[4]>);
static_assert(!AcceptsGatherRange<std::array<std::uint64_t, 4>>);
static_assert(!AcceptsGatherRange<std::vector<bool>>);
static_assert(!AcceptsGatherRange<std::initializer_list<std::uint32_t>>);
static_assert(!std::is_copy_constructible_v<IntFlow>);
static_assert(std::is_move_constructible_v<IntFlow>);
static_assert(!std::is_copy_constructible_v<DeferredIntFlow>);
static_assert(std::is_move_constructible_v<DeferredIntFlow>);
static_assert(std::is_copy_constructible_v<IntStage>);
static_assert(std::is_copy_constructible_v<HeterogeneousZip>);
static_assert(std::is_copy_constructible_v<HeterogeneousRecord>);
static_assert(
    CanZip<
        rund::compute::StageRef<std::int32_t, rund::compute::stage::Exact>,
        rund::compute::StageRef<std::uint32_t, rund::compute::stage::Exact>>);
static_assert(
    !CanZip<
        rund::compute::StageRef<std::int32_t, rund::compute::stage::Exact>,
        rund::compute::StageRef<std::int64_t, rund::compute::stage::Exact>>);
static_assert(
    !CanZip<
        rund::compute::StageRef<std::int32_t, rund::compute::stage::Exact>,
        rund::compute::StageRef<std::int32_t, rund::compute::stage::Scalar>>);

} // namespace rund_node_flow_contract
