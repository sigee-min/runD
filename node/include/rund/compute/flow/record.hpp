#pragma once

#include <rund/compute/flow/branch.hpp>

namespace rund::compute {
template <class... Schema, class... A>
class Flow<Record<Schema...>(A...), stage::Exact, input::Bound> final {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

  template <class Fn> [[nodiscard]] auto branch(Fn &&function) && {
    auto selected =
        std::forward<Fn>(function)(RecordRef<Schema...>{state_, values_});
    using Selected = std::remove_cvref_t<decltype(selected)>;
    static_assert(
        detail::is_selection<Selected> || detail::is_record<Selected> ||
            detail::is_stage_ref<Selected>,
        "compute record branch must return a stage, record, or outputs");
    if constexpr (detail::is_selection<Selected>) {
      state_.reset();
      return std::move(selected).template bind<A...>();
    } else if constexpr (detail::is_record<Selected>) {
      auto terminal = outputs(selected);
      state_.reset();
      return std::move(terminal).template bind<A...>();
    } else {
      using U = typename Selected::Value;
      using Card = typename Selected::Cardinality;
      if constexpr (detail::is_bounded_stage<Card>) {
        return Flow<U(A...), Card>{std::move(state_), selected.value_,
                                   selected.count_};
      } else {
        detail::flow_pick(state_, selected.value_);
        return Flow<U(A...), Card>{std::move(state_)};
      }
    }
  }

  [[nodiscard]] auto compile_async() && {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Outputs<Schema...>(A...)>> compile() && {
    detail::flow_outputs(state_, values_);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Outputs<Schema...>(A...)>>::fail(compiled.reason());
    }
    return Result<Program<Outputs<Schema...>(A...)>>::success(
        Program<Outputs<Schema...>(A...)>{std::move(compiled).value()});
  }

  [[nodiscard]] Result<std::tuple<detail::HostValueT<Schema>...>> collect() && {
    detail::flow_outputs(state_, values_);
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::tuple<detail::HostValueT<Schema>...>>::fail(
          compiled.reason());
    }
    return detail::run_host_outputs<Schema...>(std::move(compiled).value(),
                                               detail::flow_bindings(recipe));
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state,
       const std::array<std::uint32_t,
                        detail::schema_leaf_count<Record<Schema...>>>
           values)
      : state_(std::move(state)), values_(values) {}
  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, detail::schema_leaf_count<Record<Schema...>>>
      values_{};
};

template <class... Schema, class... A>
class Flow<Record<Schema...>(A...), stage::Exact, input::Deferred> final {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

  template <class Fn> [[nodiscard]] auto branch(Fn &&function) && {
    auto selected =
        std::forward<Fn>(function)(RecordRef<Schema...>{state_, values_});
    using Selected = std::remove_cvref_t<decltype(selected)>;
    static_assert(
        detail::is_selection<Selected> || detail::is_record<Selected> ||
            detail::is_stage_ref<Selected>,
        "compute record branch must return a stage, record, or outputs");
    if constexpr (detail::is_selection<Selected>) {
      state_.reset();
      return std::move(selected).template bind<A...>();
    } else if constexpr (detail::is_record<Selected>) {
      auto terminal = outputs(selected);
      state_.reset();
      return std::move(terminal).template bind<A...>();
    } else {
      using U = typename Selected::Value;
      using Card = typename Selected::Cardinality;
      if constexpr (detail::is_bounded_stage<Card>) {
        return detail::DeferredFlow<U(A...), Card>{
            std::move(state_), selected.value_, selected.count_};
      } else {
        detail::flow_pick(state_, selected.value_);
        return detail::DeferredFlow<U(A...), Card>{std::move(state_)};
      }
    }
  }

  [[nodiscard]] auto compile_async() && {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Outputs<Schema...>(A...)>> compile() && {
    detail::flow_outputs(state_, values_);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Outputs<Schema...>(A...)>>::fail(compiled.reason());
    }
    return Result<Program<Outputs<Schema...>(A...)>>::success(
        Program<Outputs<Schema...>(A...)>{std::move(compiled).value()});
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state,
       const std::array<std::uint32_t,
                        detail::schema_leaf_count<Record<Schema...>>>
           values)
      : state_(std::move(state)), values_(values) {}
  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, detail::schema_leaf_count<Record<Schema...>>>
      values_{};
};

} // namespace rund::compute
