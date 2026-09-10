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
} // namespace rund::compute
