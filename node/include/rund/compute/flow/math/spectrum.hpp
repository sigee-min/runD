#pragma once

#include <rund/compute/flow.hpp>

namespace rund::compute {
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
