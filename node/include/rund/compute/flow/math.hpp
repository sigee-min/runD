#pragma once

#include <rund/compute/flow.hpp>

namespace rund::compute {
template <class R, class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<R(A...), stage::Complex, Inputs> final
    : public detail::StagePipe<Flow<R(A...), stage::Complex, Inputs>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
  [[nodiscard]] Flow &&fourier(const Transform options = {}) && {
    const std::size_t count = detail::flow_value_count(state_, real_);
    if (count == 0u || (count & (count - 1u)) != 0u) {
      detail::flow_reject(state_, Reason::TransformCountNotPowerOfTwo);
      return std::move(*this);
    }
    const detail::ComplexIds output = detail::flow_transform(
        state_, real_, imag_,
        {.mode = static_cast<std::uint32_t>(options.direction),
         .flag = options.normalize});
    real_ = output.real;
    imag_ = output.imag;
    return std::move(*this);
  }
  [[nodiscard]] Flow<R(A...), stage::Exact, Inputs> real() && {
    detail::flow_pick(state_, real_);
    return Flow<R(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  [[nodiscard]] Flow<R(A...), stage::Exact, Inputs> imag() && {
    detail::flow_pick(state_, imag_);
    return Flow<R(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  [[nodiscard]] Result<std::tuple<std::vector<R>, std::vector<R>>> collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    const std::array<std::uint32_t, 2u> selected{real_, imag_};
    detail::flow_outputs(state_, selected);
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::tuple<std::vector<R>, std::vector<R>>>::fail(
          compiled.reason());
    }
    return detail::run_host_outputs<R, R>(std::move(compiled).value(),
                                          detail::flow_bindings(recipe));
  }
  [[nodiscard]] auto compile_async() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Outputs<R, R>(A...)>> compile() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    const std::array<std::uint32_t, 2u> selected{real_, imag_};
    detail::flow_outputs(state_, selected);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Outputs<R, R>(A...)>>::fail(compiled.reason());
    }
    return Result<Program<Outputs<R, R>(A...)>>::success(
        Program<Outputs<R, R>(A...)>{std::move(compiled).value()});
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state, const std::uint32_t real,
       const std::uint32_t imag)
      : state_(std::move(state)), real_(real), imag_(imag) {}
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t real_{};
  std::uint32_t imag_{};
};
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
template <class R, std::size_t Rows, std::size_t Cols, std::size_t Batches,
          class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<R(A...), stage::Solve<Rows, Cols, Batches>, Inputs> final
    : public detail::StagePipe<
          Flow<R(A...), stage::Solve<Rows, Cols, Batches>, Inputs>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
  [[nodiscard]] Flow<R(A...), stage::Matrix<Rows, Cols, Batches>, Inputs>
  values() && {
    detail::flow_pick(state_, values_);
    return {std::move(state_), shape_};
  }
  [[nodiscard]] Flow<std::uint32_t(A...), stage::Exact, Inputs> status() && {
    detail::flow_pick(state_, status_);
    return Flow<std::uint32_t(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  [[nodiscard]] Result<std::tuple<std::vector<R>, std::vector<std::uint32_t>>>
  collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    const std::array<std::uint32_t, 2u> selected{values_, status_};
    detail::flow_outputs(state_, selected);
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::tuple<std::vector<R>, std::vector<std::uint32_t>>>::
          fail(compiled.reason());
    }
    return detail::run_host_outputs<R, std::uint32_t>(
        std::move(compiled).value(), detail::flow_bindings(recipe));
  }
  [[nodiscard]] auto compile_async() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Outputs<R, std::uint32_t>(A...)>> compile() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    const std::array<std::uint32_t, 2u> selected{values_, status_};
    detail::flow_outputs(state_, selected);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Outputs<R, std::uint32_t>(A...)>>::fail(
          compiled.reason());
    }
    return Result<Program<Outputs<R, std::uint32_t>(A...)>>::success(
        Program<Outputs<R, std::uint32_t>(A...)>{std::move(compiled).value()});
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state, const std::uint32_t values,
       const std::uint32_t status, const MatrixShape shape)
      : state_(std::move(state)), values_(values), status_(status),
        shape_(shape) {}
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t values_{};
  std::uint32_t status_{};
  MatrixShape shape_{};
};
template <class R, SpectrumOp Op, SpectrumVectors V, std::size_t Rows,
          std::size_t Cols, std::size_t Batches, class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<R(A...), stage::Spectrum<Op, V, Rows, Cols, Batches>, Inputs> final
    : public detail::StagePipe<
          Flow<R(A...), stage::Spectrum<Op, V, Rows, Cols, Batches>, Inputs>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
  [[nodiscard]] Flow<R(A...), stage::Exact, Inputs> values() && {
    detail::flow_pick(state_, values_);
    return Flow<R(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  [[nodiscard]] auto vectors() &&
    requires(V != SpectrumVectors::Values)
  {
    detail::flow_pick(state_, vectors_);
    constexpr std::size_t width =
        Rows == stage::Dynamic || Cols == stage::Dynamic
            ? stage::Dynamic
            : (Rows < Cols ? Rows : Cols);
    constexpr std::size_t vector_cols =
        V == SpectrumVectors::Thin ? width : Rows;
    const std::size_t runtime_width =
        Op == SpectrumOp::Svd
            ? (shape_.rows < shape_.cols ? shape_.rows : shape_.cols)
            : shape_.rows;
    const MatrixShape vectors_shape{
        shape_.rows, V == SpectrumVectors::Thin ? runtime_width : shape_.rows,
        shape_.batches};
    return Flow<R(A...), stage::Matrix<Rows, vector_cols, Batches>, Inputs>{
        std::move(state_), vectors_shape};
  }
  [[nodiscard]] Flow<std::uint32_t(A...), stage::Exact, Inputs> status() && {
    detail::flow_pick(state_, status_);
    return Flow<std::uint32_t(A...), stage::Exact, Inputs>{std::move(state_)};
  }
  [[nodiscard]] auto collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    auto recipe = std::move(state_);
    if constexpr (V == SpectrumVectors::Values) {
      const std::array<std::uint32_t, 2u> selected{values_, status_};
      detail::flow_outputs(recipe, selected);
      auto compiled = detail::compile_flow(recipe);
      if (!compiled) {
        return Result<std::tuple<std::vector<R>, std::vector<std::uint32_t>>>::
            fail(compiled.reason());
      }
      return detail::run_host_outputs<R, std::uint32_t>(
          std::move(compiled).value(), detail::flow_bindings(recipe));
    } else {
      const std::array<std::uint32_t, 3u> selected{values_, vectors_, status_};
      detail::flow_outputs(recipe, selected);
      auto compiled = detail::compile_flow(recipe);
      if (!compiled) {
        return Result<
            std::tuple<std::vector<R>, std::vector<R>,
                       std::vector<std::uint32_t>>>::fail(compiled.reason());
      }
      return detail::run_host_outputs<R, R, std::uint32_t>(
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
    if constexpr (V == SpectrumVectors::Values) {
      const std::array<std::uint32_t, 2u> selected{values_, status_};
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
    } else {
      const std::array<std::uint32_t, 3u> selected{values_, vectors_, status_};
      detail::flow_outputs(state_, selected);
      auto compiled = detail::compile_flow(state_);
      state_.reset();
      if (!compiled) {
        return Result<Program<Outputs<R, R, std::uint32_t>(A...)>>::fail(
            compiled.reason());
      }
      return Result<Program<Outputs<R, R, std::uint32_t>(A...)>>::success(
          Program<Outputs<R, R, std::uint32_t>(A...)>{
              std::move(compiled).value()});
    }
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state, const std::uint32_t values,
       const std::uint32_t vectors, const std::uint32_t status,
       const MatrixShape shape)
      : state_(std::move(state)), values_(values), vectors_(vectors),
        status_(status), shape_(shape) {}
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t values_{};
  std::uint32_t vectors_{};
  std::uint32_t status_{};
  MatrixShape shape_{};
};
} // namespace rund::compute
