#pragma once

#include <rund/compute/flow.hpp>

namespace rund::compute {
template <class R, FactorOp Op, std::size_t Rows, std::size_t Cols,
          std::size_t Batches, class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<R(A...), stage::Factor<Op, Rows, Cols, Batches>, Inputs> final
    : public detail::StagePipe<
          Flow<R(A...), stage::Factor<Op, Rows, Cols, Batches>, Inputs>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
  [[nodiscard]] auto packed() && {
    detail::flow_pick(state_, packed_);
    if constexpr (Op == FactorOp::Qr) {
      return Flow<R(A...), stage::Exact, Inputs>{std::move(state_)};
    } else {
      return Flow<R(A...), stage::Matrix<Rows, Cols, Batches>, Inputs>{
          std::move(state_), shape_};
    }
  }
  [[nodiscard]] Flow<std::uint32_t(A...), stage::Exact, Inputs> status() && {
    detail::flow_pick(state_, status_);
    return Flow<std::uint32_t(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  [[nodiscard]] Flow<std::uint32_t(A...), stage::Exact, Inputs> pivots() &&
    requires(Op == FactorOp::Lu)
  {
    detail::flow_pick(state_, pivots_);
    return Flow<std::uint32_t(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  template <class Range>
    requires(std::same_as<Inputs, input::Bound> &&
             detail::BorrowedRange<Range, R>)
  [[nodiscard]] Flow<R(A..., R), stage::Solve<Rows, stage::Dynamic, Batches>,
                     Inputs>
  solve(Range &range, const std::size_t rhs_cols = 1u) && {
    const std::span<const R> rhs{range};
    const std::uint32_t side =
        detail::flow_side(state_, {rhs.data(), rhs.size(), detail::type<R>()},
                          true, detail::storage_format<R>());
    const detail::SolveIds result =
        detail::flow_factor_solve(state_, Op, packed_, pivots_, side,
                                  shape_.rows, rhs_cols, shape_.batches);
    return {std::move(state_), result.values, result.status,
            MatrixShape{shape_.rows, rhs_cols, shape_.batches}};
  }
  template <std::size_t RhsCols, class Range>
    requires(std::same_as<Inputs, input::Bound> && RhsCols != 0u &&
             detail::BorrowedRange<Range, R>)
  [[nodiscard]] Flow<R(A..., R), stage::Solve<Rows, RhsCols, Batches>, Inputs>
  solve(Range &range) && {
    const std::span<const R> rhs{range};
    const std::uint32_t side =
        detail::flow_side(state_, {rhs.data(), rhs.size(), detail::type<R>()},
                          true, detail::storage_format<R>());
    const detail::SolveIds result =
        detail::flow_factor_solve(state_, Op, packed_, pivots_, side,
                                  shape_.rows, RhsCols, shape_.batches);
    return {std::move(state_), result.values, result.status,
            MatrixShape{shape_.rows, RhsCols, shape_.batches}};
  }
  [[nodiscard]] Flow<R(A..., R), stage::Solve<Rows, stage::Dynamic, Batches>,
                     Inputs>
  solve(const std::size_t rhs_cols = 1u) &&
    requires std::same_as<Inputs, input::Deferred>
  {
    const std::size_t rhs_count = detail::flow_matrix_extent(
        state_, shape_.rows, rhs_cols, shape_.batches);
    const std::uint32_t side =
        detail::flow_side(state_, {nullptr, rhs_count, detail::type<R>()},
                          false, detail::storage_format<R>());
    const detail::SolveIds result =
        detail::flow_factor_solve(state_, Op, packed_, pivots_, side,
                                  shape_.rows, rhs_cols, shape_.batches);
    return {std::move(state_), result.values, result.status,
            MatrixShape{shape_.rows, rhs_cols, shape_.batches}};
  }
  template <std::size_t RhsCols>
  [[nodiscard]] Flow<R(A..., R), stage::Solve<Rows, RhsCols, Batches>, Inputs>
  solve() &&
    requires(std::same_as<Inputs, input::Deferred> && RhsCols != 0u)
  {
    const std::size_t rhs_count = detail::flow_matrix_extent(
        state_, shape_.rows, RhsCols, shape_.batches);
    const std::uint32_t side =
        detail::flow_side(state_, {nullptr, rhs_count, detail::type<R>()},
                          false, detail::storage_format<R>());
    const detail::SolveIds result =
        detail::flow_factor_solve(state_, Op, packed_, pivots_, side,
                                  shape_.rows, RhsCols, shape_.batches);
    return {std::move(state_), result.values, result.status,
            MatrixShape{shape_.rows, RhsCols, shape_.batches}};
  }
  [[nodiscard]] auto collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    auto recipe = std::move(state_);
    if constexpr (Op == FactorOp::Lu) {
      const std::array<std::uint32_t, 3u> selected{packed_, pivots_, status_};
      detail::flow_outputs(recipe, selected);
      auto compiled = detail::compile_flow(recipe);
      if (!compiled) {
        return Result<
            std::tuple<std::vector<R>, std::vector<std::uint32_t>,
                       std::vector<std::uint32_t>>>::fail(compiled.reason());
      }
      return detail::run_host_outputs<R, std::uint32_t, std::uint32_t>(
          std::move(compiled).value(), detail::flow_bindings(recipe));
    } else {
      const std::array<std::uint32_t, 2u> selected{packed_, status_};
      detail::flow_outputs(recipe, selected);
      auto compiled = detail::compile_flow(recipe);
      if (!compiled) {
        return Result<std::tuple<std::vector<R>, std::vector<std::uint32_t>>>::
            fail(compiled.reason());
      }
      return detail::run_host_outputs<R, std::uint32_t>(
          std::move(compiled).value(), detail::flow_bindings(recipe));
    }
  }
  [[nodiscard]] auto compile_async() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] auto compile() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    if constexpr (Op == FactorOp::Lu) {
      const std::array<std::uint32_t, 3u> selected{packed_, pivots_, status_};
      detail::flow_outputs(state_, selected);
      auto compiled = detail::compile_flow(state_);
      state_.reset();
      if (!compiled) {
        return Result<Program<Outputs<R, std::uint32_t, std::uint32_t>(A...)>>::
            fail(compiled.reason());
      }
      return Result<Program<Outputs<R, std::uint32_t, std::uint32_t>(A...)>>::
          success(Program<Outputs<R, std::uint32_t, std::uint32_t>(A...)>{
              std::move(compiled).value()});
    } else {
      const std::array<std::uint32_t, 2u> selected{packed_, status_};
      detail::flow_outputs(state_, selected);
      auto compiled = detail::compile_flow(state_);
      state_.reset();
      if (!compiled) {
        return Result<Program<Outputs<R, std::uint32_t>(A...)>>::fail(
            compiled.reason());
      }
      return Result<Program<Outputs<R, std::uint32_t>(A...)>>::success(
          Program<Outputs<R, std::uint32_t>(A...)>{
              std::move(compiled).value()});
    }
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state, const std::uint32_t packed,
       const std::uint32_t pivots, const std::uint32_t status,
       const MatrixShape shape)
      : state_(std::move(state)), packed_(packed), pivots_(pivots),
        status_(status), shape_(shape) {}
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t packed_{};
  std::uint32_t pivots_{};
  std::uint32_t status_{};
  MatrixShape shape_{};
};
} // namespace rund::compute
