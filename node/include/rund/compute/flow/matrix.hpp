#pragma once

#include <rund/compute/flow.hpp>

namespace rund::compute {
template <class R, std::size_t Rows, std::size_t Cols, std::size_t Batches,
          class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<R(A...), stage::Matrix<Rows, Cols, Batches>, Inputs> final
    : public detail::StagePipe<
          Flow<R(A...), stage::Matrix<Rows, Cols, Batches>, Inputs>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
  [[nodiscard]] auto transpose() && {
    detail::flow_unary(
        state_, detail::Primitive::Matrix, detail::type<R>(),
        detail::flow_output_count(state_),
        {.first = shape_.rows,
         .second = shape_.cols,
         .fourth = shape_.batches,
         .mode = static_cast<std::uint32_t>(detail::MatrixMode::Transpose)});
    const MatrixShape shape{shape_.cols, shape_.rows, shape_.batches};
    return Flow<R(A...), stage::Matrix<Cols, Rows, Batches>, Inputs>{
        std::move(state_), shape};
  }

  template <class Range>
    requires(std::same_as<Inputs, input::Bound> &&
             detail::BorrowedRange<Range, R>)
  [[nodiscard]] auto matmul(Range &range, const MatrixShape right) && {
    const std::span<const R> values{range};
    const std::size_t output_count = detail::flow_matrix_product(
        state_, shape_.rows, shape_.cols, shape_.batches, right.rows,
        right.cols, right.batches, values.size());
    append_product(values.data(), values.size(), output_count, right.cols,
                   true);
    return Flow<R(A..., R), stage::Matrix<Rows, stage::Dynamic, Batches>,
                Inputs>{std::move(state_),
                        MatrixShape{shape_.rows, right.cols, shape_.batches}};
  }

  template <std::size_t RightRows, std::size_t RightCols,
            std::size_t RightBatches = 1u, class Range>
    requires(std::same_as<Inputs, input::Bound> &&
             (Cols == stage::Dynamic || Cols == RightRows) &&
             (Batches == stage::Dynamic || Batches == RightBatches) &&
             RightRows != 0u && RightCols != 0u && RightBatches != 0u &&
             detail::BorrowedRange<Range, R>)
  [[nodiscard]] auto matmul(Range &range) && {
    const std::span<const R> values{range};
    const std::size_t output_count = detail::flow_matrix_product(
        state_, shape_.rows, shape_.cols, shape_.batches, RightRows, RightCols,
        RightBatches, values.size());
    append_product(values.data(), values.size(), output_count, RightCols, true);
    return Flow<R(A..., R), stage::Matrix<Rows, RightCols, Batches>, Inputs>{
        std::move(state_), MatrixShape{shape_.rows, RightCols, shape_.batches}};
  }

  [[nodiscard]] auto matmul(const MatrixShape right) &&
    requires std::same_as<Inputs, input::Deferred>
  {
    return product<stage::Dynamic>(right);
  }

  template <std::size_t RightRows, std::size_t RightCols,
            std::size_t RightBatches = 1u>
    requires(std::same_as<Inputs, input::Deferred> &&
             (Cols == stage::Dynamic || Cols == RightRows) &&
             (Batches == stage::Dynamic || Batches == RightBatches) &&
             RightRows != 0u && RightCols != 0u && RightBatches != 0u)
  [[nodiscard]] auto matmul() && {
    return product<RightCols>(MatrixShape{RightRows, RightCols, RightBatches});
  }

  template <class Range>
    requires(detail::FixedValue<R> && std::same_as<Inputs, input::Bound> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic ||
              Rows == Cols) &&
             detail::BorrowedRange<Range, R>)
  [[nodiscard]] auto solve(Range &range, const FactorOp operation,
                           const std::size_t rhs_cols = 1u) && {
    const std::span<const R> rhs{range};
    return solve_with<stage::Dynamic>(operation, rhs.data(), rhs.size(),
                                      rhs_cols, true);
  }

  template <FactorOp Op, std::size_t RhsCols, class Range>
    requires(detail::FixedValue<R> && std::same_as<Inputs, input::Bound> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic ||
              Rows == Cols) &&
             (Op == FactorOp::Lu || Op == FactorOp::Qr ||
              Op == FactorOp::Cholesky) &&
             RhsCols != 0u && detail::BorrowedRange<Range, R>)
  [[nodiscard]] auto solve(Range &range) && {
    const std::span<const R> rhs{range};
    return solve_with<RhsCols>(Op, rhs.data(), rhs.size(), RhsCols, true);
  }

  [[nodiscard]] auto solve(const FactorOp operation,
                           const std::size_t rhs_cols = 1u) &&
    requires(detail::FixedValue<R> && std::same_as<Inputs, input::Deferred> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic || Rows == Cols))
  {
    const std::size_t rhs_count = detail::flow_matrix_extent(
        state_, shape_.rows, rhs_cols, shape_.batches);
    return solve_with<stage::Dynamic>(operation, nullptr, rhs_count, rhs_cols,
                                      false);
  }

  template <FactorOp Op, std::size_t RhsCols>
    requires(detail::FixedValue<R> && std::same_as<Inputs, input::Deferred> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic ||
              Rows == Cols) &&
             (Op == FactorOp::Lu || Op == FactorOp::Qr ||
              Op == FactorOp::Cholesky) &&
             RhsCols != 0u)
  [[nodiscard]] auto solve() && {
    const std::size_t rhs_count = detail::flow_matrix_extent(
        state_, shape_.rows, RhsCols, shape_.batches);
    return solve_with<RhsCols>(Op, nullptr, rhs_count, RhsCols, false);
  }

  [[nodiscard]] auto lu() &&
    requires(detail::FixedValue<R> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic || Rows == Cols))
  {
    return factor<FactorOp::Lu>();
  }

  [[nodiscard]] auto qr() &&
    requires detail::FixedValue<R>
  {
    return factor<FactorOp::Qr>();
  }

  [[nodiscard]] auto cholesky() &&
    requires(detail::FixedValue<R> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic || Rows == Cols))
  {
    return factor<FactorOp::Cholesky>();
  }
  template <SpectrumVectors V = SpectrumVectors::Values>
  [[nodiscard]] auto svd(const std::uint32_t iterations = 32u) &&
    requires detail::FixedValue<R>
  {
    return spectrum<SpectrumOp::Svd, V>(iterations);
  }
  template <SpectrumVectors V = SpectrumVectors::Values>
    requires(detail::FixedValue<R> &&
             (Rows == stage::Dynamic || Cols == stage::Dynamic || Rows == Cols))
  [[nodiscard]] auto eigen(const std::uint32_t iterations = 32u) && {
    return spectrum<SpectrumOp::Eigen, V>(iterations);
  }

  [[nodiscard]] Result<std::vector<R>> collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::vector<R>>::fail(compiled.reason());
    }
    return detail::run_host_views<R>(std::move(compiled).value(),
                                     detail::flow_bindings(recipe));
  }
  [[nodiscard]] auto compile_async() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<R(A...)>> compile() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<R(A...)>>::fail(compiled.reason());
    }
    return Result<Program<R(A...)>>::success(
        Program<R(A...)>{std::move(compiled).value()});
  }

private:
  friend class detail::StagePipe<
      Flow<R(A...), stage::Matrix<Rows, Cols, Batches>, Inputs>>;
  template <class, class, class> friend class Flow;
  template <class Fn> [[nodiscard]] auto pipe_stage(Fn &&function) && {
    auto next = std::forward<Fn>(function)(
        StageRef<R, stage::Matrix<Rows, Cols, Batches>>{
            state_, detail::flow_value(state_), 0u, shape_});
    using Next = std::remove_cvref_t<decltype(next)>;
    static_assert(detail::is_stage_ref<Next>,
                  "compute matrix pipe must return a typed stage");
    if (state_ != detail::StageRefAccess::state(next)) {
      detail::flow_pick(state_, 0u);
    }
    using U = typename Next::Value;
    using Card = typename Next::Cardinality;
    if constexpr (detail::is_solve_stage<Card>) {
      return Flow<U(A...), Card, Inputs>{std::move(state_), next.value_,
                                         next.count_, next.shape_};
    } else if constexpr (detail::is_matrix_stage<Card>) {
      detail::flow_pick(state_, next.value_);
      return Flow<U(A...), Card, Inputs>{std::move(state_), next.shape_};
    } else if constexpr (detail::is_bounded_stage<Card>) {
      return Flow<U(A...), Card, Inputs>{std::move(state_), next.value_,
                                         next.count_};
    } else {
      detail::flow_pick(state_, next.value_);
      return Flow<U(A...), Card, Inputs>{std::move(state_)};
    }
  }
  template <std::size_t RhsCols>
  [[nodiscard]] auto solve_with(const FactorOp operation, const R *const data,
                                const std::size_t count,
                                const std::size_t rhs_cols, const bool bind) {
    const std::uint32_t side =
        detail::flow_side(state_, {data, count, detail::type<R>()}, bind,
                          detail::storage_format<R>());
    const detail::SolveIds result = detail::flow_matrix_solve(
        state_, operation, detail::flow_value(state_), side, shape_.rows,
        shape_.cols, rhs_cols, shape_.batches);
    return Flow<R(A..., R), stage::Solve<Rows, RhsCols, Batches>, Inputs>{
        std::move(state_), result.values, result.status,
        MatrixShape{shape_.rows, rhs_cols, shape_.batches}};
  }
  template <std::size_t RightCols>
  [[nodiscard]] auto product(const MatrixShape right) {
    const std::size_t right_count = detail::flow_matrix_extent(
        state_, right.rows, right.cols, right.batches);
    const std::size_t output_count = detail::flow_matrix_product(
        state_, shape_.rows, shape_.cols, shape_.batches, right.rows,
        right.cols, right.batches, right_count);
    append_product(nullptr, right_count, output_count, right.cols, false);
    return Flow<R(A..., R), stage::Matrix<Rows, RightCols, Batches>, Inputs>{
        std::move(state_),
        MatrixShape{shape_.rows, right.cols, shape_.batches}};
  }
  void append_product(const R *const data, const std::size_t count,
                      const std::size_t output_count,
                      const std::size_t right_cols, const bool bind) {
    const std::uint32_t side =
        detail::flow_side(state_, {data, count, detail::type<R>()}, bind,
                          detail::storage_format<R>());
    detail::flow_binary(
        state_, detail::Primitive::Matrix, side, false, detail::type<R>(),
        output_count,
        {.first = shape_.rows,
         .second = right_cols,
         .third = shape_.cols,
         .fourth = shape_.batches,
         .mode = static_cast<std::uint32_t>(
             shape_.batches == 1u ? detail::MatrixMode::Mul
                                  : detail::MatrixMode::BatchMul)});
  }
  template <FactorOp Op> [[nodiscard]] auto factor() {
    const detail::FactorIds result = detail::flow_factor(
        state_, Op, shape_.rows, shape_.cols, shape_.batches);
    return Flow<R(A...), stage::Factor<Op, Rows, Cols, Batches>, Inputs>{
        std::move(state_), result.packed, result.pivots, result.status, shape_};
  }
  template <SpectrumOp Op, SpectrumVectors V>
  [[nodiscard]] auto spectrum(const std::uint32_t iterations) {
    const detail::SpectrumIds result = detail::flow_spectrum(
        state_, Op, V, shape_.rows, shape_.cols, shape_.batches, iterations);
    return Flow<R(A...), stage::Spectrum<Op, V, Rows, Cols, Batches>, Inputs>{
        std::move(state_), result.values, result.vectors, result.status,
        shape_};
  }
  Flow(std::shared_ptr<detail::FlowState> state, const MatrixShape shape)
      : state_(std::move(state)), shape_(shape) {}
  std::shared_ptr<detail::FlowState> state_;
  MatrixShape shape_{};
};
} // namespace rund::compute
