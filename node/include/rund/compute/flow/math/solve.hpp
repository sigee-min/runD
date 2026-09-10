#pragma once

#include <rund/compute/flow.hpp>

namespace rund::compute {
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
} // namespace rund::compute
