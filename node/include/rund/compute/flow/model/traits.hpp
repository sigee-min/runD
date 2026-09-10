#pragma once

#include <rund/compute/flow/model/base.hpp>

namespace rund::compute::detail {

struct StageRefAccess final {
  template <class T, class Card>
  [[nodiscard]] static StageRef<T, Card> make(std::shared_ptr<FlowState> state,
                                              const std::uint32_t value,
                                              const std::uint32_t count = 0u) {
    return StageRef<T, Card>{std::move(state), value, count};
  }
  template <class T, class Card>
  [[nodiscard]] static const std::shared_ptr<FlowState> &
  state(const StageRef<T, Card> &value) noexcept {
    return value.state_;
  }
  template <class T, class Card>
  [[nodiscard]] static std::uint32_t
  id(const StageRef<T, Card> &value) noexcept {
    return value.value_;
  }
  template <class T, class Card>
  [[nodiscard]] static std::uint32_t
  count(const StageRef<T, Card> &value) noexcept {
    return value.count_;
  }
};
template <class T> inline constexpr bool is_bounded_stage = false;
template <class Count>
inline constexpr bool is_bounded_stage<stage::Bounded<Count>> = true;
struct NoSide final {};
template <class T, class Card> struct ResidentCount final {
  using Type = CountFor<T>;
};
template <class T, class Count>
struct ResidentCount<T, stage::Bounded<Count>> final {
  using Type = Count;
};
template <class T, class Card>
using ResidentCountT = typename ResidentCount<T, Card>::Type;
template <class T> inline constexpr bool is_matrix_stage = false;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
inline constexpr bool is_matrix_stage<stage::Matrix<Rows, Cols, Batches>> =
    true;
template <class T> inline constexpr bool is_solve_stage = false;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
inline constexpr bool is_solve_stage<stage::Solve<Rows, Cols, Batches>> = true;
template <class T> struct MatrixStageTraits;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
struct MatrixStageTraits<stage::Matrix<Rows, Cols, Batches>> final {
  inline static constexpr std::size_t rows = Rows;
  inline static constexpr std::size_t cols = Cols;
  inline static constexpr std::size_t batches = Batches;
};
template <class T> struct SolveStageTraits;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
struct SolveStageTraits<stage::Solve<Rows, Cols, Batches>> final {
  inline static constexpr std::size_t rows = Rows;
  inline static constexpr std::size_t cols = Cols;
  inline static constexpr std::size_t batches = Batches;
};
template <class T>
inline constexpr bool is_element_stage =
    std::same_as<T, stage::Exact> || std::same_as<T, stage::Scalar> ||
    is_bounded_stage<T>;
template <class T>
inline constexpr bool is_matrix_solve_stage = [] {
  if constexpr (!is_matrix_stage<T>) {
    return false;
  } else {
    using Matrix = MatrixStageTraits<T>;
    return Matrix::rows == stage::Dynamic || Matrix::cols == stage::Dynamic ||
           Matrix::rows == Matrix::cols;
  }
}();
template <class MatrixCard, class RhsCard>
inline constexpr bool matrix_rhs_stage_compatible = [] {
  if constexpr (!is_matrix_stage<MatrixCard>) {
    return false;
  } else if constexpr (std::same_as<RhsCard, stage::Exact>) {
    return true;
  } else if constexpr (is_matrix_stage<RhsCard>) {
    using Left = MatrixStageTraits<MatrixCard>;
    using Right = MatrixStageTraits<RhsCard>;
    return (Left::rows == stage::Dynamic || Right::rows == stage::Dynamic ||
            Left::rows == Right::rows) &&
           (Left::batches == stage::Dynamic ||
            Right::batches == stage::Dynamic ||
            Left::batches == Right::batches);
  } else {
    return false;
  }
}();
template <class MatrixCard, class RhsCard, std::size_t RhsCols>
inline constexpr bool matrix_rhs_static_compatible = [] {
  if constexpr (!matrix_rhs_stage_compatible<MatrixCard, RhsCard>) {
    return false;
  } else if constexpr (std::same_as<RhsCard, stage::Exact>) {
    return true;
  } else {
    using Right = MatrixStageTraits<RhsCard>;
    return Right::cols == stage::Dynamic || Right::cols == RhsCols;
  }
}();
template <class T>
concept IntegerValue =
    std::same_as<T, std::int32_t> || std::same_as<T, std::uint32_t> ||
    std::same_as<T, std::int64_t> || std::same_as<T, std::uint64_t>;
template <class Card>
[[nodiscard]] inline FlowControl
stage_control(const std::shared_ptr<FlowState> &state,
              const std::uint32_t value, const std::uint32_t count) {
  if constexpr (is_bounded_stage<Card>) {
    return FlowControl{.count = count,
                       .capacity = flow_value_count(state, value)};
  } else {
    return {};
  }
}
template <class T> inline constexpr bool is_initializer_list = false;
template <class T>
inline constexpr bool is_initializer_list<std::initializer_list<T>> = true;
template <class Range, class = void> struct BorrowedRangeTraits final {};
template <class Range>
struct BorrowedRangeTraits<
    Range, std::void_t<decltype(std::span{std::declval<Range &>()})>>
    final {
  using Span = decltype(std::span{std::declval<Range &>()});
  using Value = std::remove_cv_t<typename Span::element_type>;
};
template <class Range, class T>
concept BorrowedRange =
    ComputeValue<std::remove_cv_t<T>> &&
    (!is_initializer_list<std::remove_cv_t<Range>>) &&
    requires { typename BorrowedRangeTraits<Range>::Value; } &&
    std::same_as<typename BorrowedRangeTraits<Range>::Value,
                 std::remove_cv_t<T>> &&
    requires(Range &range) { std::span<const std::remove_cv_t<T>>{range}; };
template <class Range>
concept ComputeRange = requires {
  typename BorrowedRangeTraits<Range>::Value;
} && BorrowedRange<Range, typename BorrowedRangeTraits<Range>::Value>;
template <class Range>
using BorrowedRangeValue = typename BorrowedRangeTraits<Range>::Value;

} // namespace rund::compute::detail
