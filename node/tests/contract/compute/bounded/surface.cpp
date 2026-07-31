#include "model.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rund_node_bounded_contract {

template <class T>
concept CanScan = requires(T value) {
  std::move(value).scan(rund::compute::Scan::InclusiveSum);
};

template <class T>
concept CanSort = requires(T value) { std::move(value).sort(); };

using FilterFlow =
    decltype(rund::compute::on(
                 rund::compute::Target::cpu(),
                 std::declval<const std::array<std::int64_t, 4> &>())
                 .filter([](auto value) { return value > 2; }));
static_assert(
    std::same_as<
        FilterFlow,
        rund::compute::Flow<std::int64_t(std::int64_t),
                            rund::compute::stage::Bounded<std::uint64_t>>>);
static_assert(CanScan<FilterFlow>);
using FilterCountFlow = decltype(std::declval<FilterFlow>().count());
static_assert(std::same_as<FilterCountFlow,
                           rund::compute::Flow<std::uint64_t(std::int64_t),
                                               rund::compute::stage::Scalar>>);
using FilterOrderFlow = decltype(std::declval<FilterFlow>().argsort());
static_assert(
    std::same_as<
        FilterOrderFlow,
        rund::compute::Flow<std::uint32_t(std::int64_t),
                            rund::compute::stage::Bounded<std::uint64_t>>>);
using RefilteredOrderFlow = decltype(std::declval<FilterOrderFlow>().filter(
    [](auto index) { return index > 1u; }));
static_assert(std::same_as<RefilteredOrderFlow, FilterOrderFlow>);
using RefilteredOrderCountFlow =
    decltype(std::declval<RefilteredOrderFlow>().count());
static_assert(std::same_as<RefilteredOrderCountFlow,
                           rund::compute::Flow<std::uint64_t(std::int64_t),
                                               rund::compute::stage::Scalar>>);
using RefilterFlow = decltype(std::declval<FilterFlow>().filter(
    [](auto value) { return value > 3; }));
static_assert(std::same_as<RefilterFlow, FilterFlow>);
using CountFlow =
    decltype(rund::compute::on(
                 rund::compute::Target::cpu(),
                 std::declval<const std::array<std::int64_t, 4> &>())
                 .count());
static_assert(
    std::same_as<CountFlow, rund::compute::Flow<std::uint64_t(std::int64_t),
                                                rund::compute::stage::Scalar>>);
static_assert(!CanScan<CountFlow>);
static_assert(!CanSort<CountFlow>);
using TypedMapFlow =
    decltype(rund::compute::on(
                 rund::compute::Target::cpu(),
                 std::declval<const std::array<std::int32_t, 4> &>())
                 .map("mask", [](auto value) {
                   return rund::compute::select<std::uint32_t>(value > 0, 1u,
                                                               0u);
                 }));
static_assert(std::same_as<TypedMapFlow,
                           rund::compute::Flow<std::uint32_t(std::int32_t),
                                               rund::compute::stage::Exact>>);
using CompactFlow =
    decltype(rund::compute::on(
                 rund::compute::Target::cpu(),
                 std::declval<const std::array<std::uint32_t, 4> &>())
                 .compact({.capacity = 3u}));
static_assert(
    std::same_as<
        CompactFlow,
        rund::compute::Flow<std::uint32_t(std::uint32_t),
                            rund::compute::stage::Bounded<std::uint32_t>>>);
using CompactCountFlow = decltype(std::declval<CompactFlow>().count());
static_assert(std::same_as<CompactCountFlow,
                           rund::compute::Flow<std::uint32_t(std::uint32_t),
                                               rund::compute::stage::Scalar>>);
using CompactStage =
    decltype(std::declval<rund::compute::StageRef<std::uint32_t>>().compact(
        {.capacity = 3u}));
static_assert(
    std::same_as<
        CompactStage,
        rund::compute::StageRef<std::uint32_t,
                                rund::compute::stage::Bounded<std::uint32_t>>>);

} // namespace rund_node_bounded_contract
