#include "pipeline.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

template <class T, std::size_t N> struct SdkContiguousInput final {
  T values[N]{};

  [[nodiscard]] constexpr T *begin() noexcept { return values; }
  [[nodiscard]] constexpr const T *begin() const noexcept { return values; }
  [[nodiscard]] constexpr T *end() noexcept { return values + N; }
  [[nodiscard]] constexpr const T *end() const noexcept { return values + N; }
  [[nodiscard]] constexpr T *data() noexcept { return values; }
  [[nodiscard]] constexpr const T *data() const noexcept { return values; }
  [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
};

using IntJob = rund::compute::Job<std::int32_t(std::int32_t)>;
using IntProgram = rund::compute::Program<std::int32_t(std::int32_t)>;
using MultiProgram =
    rund::compute::Program<std::int32_t(std::int32_t, std::uint32_t)>;
using BoundedProgram =
    rund::compute::Program<rund::compute::Bounded<std::int32_t>(std::int32_t)>;
using OutputProgram =
    rund::compute::Program<rund::compute::Outputs<std::int32_t, std::uint32_t>(
        std::int32_t)>;
using IntFlow = rund::compute::Flow<std::int32_t(std::int32_t)>;
using IntStage =
    rund::compute::StageRef<std::int32_t, rund::compute::stage::Exact>;
using UintStage =
    rund::compute::StageRef<std::uint32_t, rund::compute::stage::Exact>;
using HeterogeneousZip = decltype(rund::compute::zip(
    std::declval<const IntStage &>(), std::declval<const UintStage &>()));
using HeterogeneousRecord = decltype(rund::compute::record(
    std::declval<const IntStage &>(), std::declval<const UintStage &>()));

template <class Range>
concept AcceptsGatherRange =
    requires(IntFlow flow, Range &value) { std::move(flow).gather(value); };

static_assert(
    std::same_as<decltype(rund::compute::on(rund::compute::Target::cpu())),
                 rund::compute::FlowBuilder>);
static_assert(std::same_as<
              decltype(rund::compute::on(rund::compute::Target::cpu(),
                                         std::declval<std::int32_t (&)[4]>())),
              IntFlow>);
static_assert(std::same_as<
              decltype(rund::compute::on(
                  rund::compute::Target::cpu(),
                  std::declval<const SdkContiguousInput<std::int32_t, 4> &>())),
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
static_assert(!std::is_copy_constructible_v<IntJob>);
static_assert(std::is_move_constructible_v<IntJob>);
static_assert(HasProgramObservers<IntProgram>);
static_assert(HasProgramObservers<MultiProgram>);
static_assert(HasProgramObservers<BoundedProgram>);
static_assert(HasProgramObservers<OutputProgram>);
static_assert(sizeof(IntProgram) == sizeof(std::shared_ptr<void>));
static_assert(sizeof(MultiProgram) == sizeof(std::shared_ptr<void>));
static_assert(sizeof(BoundedProgram) == sizeof(std::shared_ptr<void>));
static_assert(sizeof(OutputProgram) == sizeof(std::shared_ptr<void>));
static_assert(!std::is_copy_constructible_v<IntFlow>);
static_assert(std::is_move_constructible_v<IntFlow>);
static_assert(std::is_copy_constructible_v<IntStage>);
static_assert(std::is_copy_constructible_v<HeterogeneousZip>);
static_assert(std::is_copy_constructible_v<HeterogeneousRecord>);
static_assert(
    std::is_same_v<decltype(std::declval<const IntProgram &>().graph()),
                   const rund::compute::graph::Info &>);
static_assert(noexcept(std::declval<const IntProgram &>().graph()));
static_assert(!AcceptsRvalueFlowInput<std::vector<std::int32_t>>);
static_assert(!AcceptsRvalueFlowInput<std::array<std::int32_t, 4>>);
static_assert(!AcceptsRvalueFlowInput<std::span<std::int32_t>>);
static_assert(AcceptsFlowRange<std::int32_t[4]>);
static_assert(AcceptsConstFlowRange<std::int32_t[4]>);
static_assert(AcceptsFlowRange<SdkContiguousInput<std::int32_t, 4>>);
static_assert(AcceptsConstFlowRange<SdkContiguousInput<std::int32_t, 4>>);
static_assert(AcceptsFlowRange<std::pmr::vector<std::int32_t>>);
static_assert(AcceptsConstFlowRange<std::pmr::vector<std::int32_t>>);
static_assert(!AcceptsFlowRange<std::initializer_list<std::int32_t>>);
static_assert(!AcceptsFlowRange<std::vector<bool>>);
static_assert(!AcceptsFlowRange<std::array<std::int16_t, 4>>);
static_assert(AcceptsGatherRange<std::uint32_t[4]>);
static_assert(AcceptsGatherRange<const std::uint32_t[4]>);
static_assert(AcceptsGatherRange<SdkContiguousInput<std::uint32_t, 4>>);
static_assert(AcceptsGatherRange<const SdkContiguousInput<std::uint32_t, 4>>);
static_assert(!AcceptsGatherRange<std::int32_t[4]>);
static_assert(!AcceptsGatherRange<std::array<std::uint64_t, 4>>);
static_assert(!AcceptsGatherRange<std::vector<bool>>);
static_assert(!AcceptsGatherRange<std::initializer_list<std::uint32_t>>);
static_assert(AcceptsFlowRange<std::vector<std::int32_t>>);
static_assert(AcceptsFlowRange<std::vector<rund::compute::Fixed<1, 31>>>);
static_assert(!AcceptsFlowRange<std::vector<float>>);
static_assert(!AcceptsFlowRange<std::vector<double>>);
static_assert(AcceptsBufferValue<std::uint64_t>);
static_assert(!AcceptsBufferValue<double>);
static_assert(!HasRemainder<rund::compute::Expr<std::int32_t>>);
static_assert(HasFixedDivide<rund::compute::Expr<rund::compute::Fixed<1, 31>>>);
static_assert(!HasFixedDivide<rund::compute::Expr<std::int32_t>>);
static_assert(!HasUnsignedStorageSaturate<rund::compute::Expr<std::int32_t>>);
static_assert(HasUnsignedStorageSaturate<rund::compute::Expr<std::uint32_t>>);
static_assert(HasUnsignedStorageSaturate<
              rund::compute::Expr<rund::compute::Fixed<16, 16>>>);
static_assert(HasUnsignedStorageSaturate<
              rund::compute::Expr<rund::compute::Fixed<20, 44>>>);
static_assert(HasArithmeticShift<rund::compute::Expr<std::int32_t>>);
static_assert(!HasArithmeticShift<rund::compute::Expr<std::uint32_t>>);
static_assert(
    HasArithmeticShift<rund::compute::Expr<rund::compute::Fixed<16, 16>>>);
