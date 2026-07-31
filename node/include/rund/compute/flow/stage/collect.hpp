#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
  template <std::size_t N, class Fn>
  [[nodiscard]] auto unroll(Fn &&function) const {
    using Next = std::remove_cvref_t<decltype(function(*this))>;
    static_assert(std::same_as<Next, StageRef>,
                  "compute unroll must preserve its typed stage");
    StageRef current = *this;
    for (std::size_t iteration = 0u; iteration < N; ++iteration) {
      current = function(current);
    }
    return current;
  }
  // Bounded solver/worklist repetition. `converged` is sampled before every
  // body and once after the last body. Converged items leave the stable active
  // set; a zero resident count therefore suppresses every later controlled
  // map and count-aware collective without a host predicate read. The result
  // is the still-active worklist, not a conditional immutable value/phi.
  template <std::size_t N, class StepFn, class ConvergedFn>
  [[nodiscard]] auto unroll(StepFn &&step,
                            ConvergedFn &&converged) const
    requires(detail::is_bounded_stage<Card>)
  {
    static_assert(N != 0u, "compute bounded unroll requires a positive bound");
    StageRef current = *this;
    for (std::size_t iteration = 0u; iteration < N; ++iteration) {
      auto keep = detail::negate_element(converged);
      auto active = current.filter(keep);
      const std::size_t first_body_step = detail::flow_step_count(state_);
      using Next = std::remove_cvref_t<decltype(step(active))>;
      static_assert(std::same_as<Next, StageRef>,
                    "compute bounded unroll step must preserve its typed "
                    "worklist");
      current = step(active);
      detail::flow_tag_iteration(
          state_, first_body_step, static_cast<std::uint32_t>(iteration + 1u));
    }
    auto keep = detail::negate_element(converged);
    return current.filter(keep);
  }
  template <class Fn>
  [[nodiscard]] auto group_by(Fn &&function) const
    requires((std::same_as<Card, stage::Exact> ||
              detail::is_bounded_stage<Card>) &&
             detail::IntegerValue<T>);
  template <class RightCard, class LeftKeyFn, class RightKeyFn, class EmitFn>
  [[nodiscard]] auto
  join(const MaxMatches bound, const StageRef<T, RightCard> &right,
       LeftKeyFn &&left_key_function, RightKeyFn &&right_key_function,
       EmitFn &&emit_function) const
    requires((std::same_as<Card, stage::Exact> ||
              detail::is_bounded_stage<Card>) &&
             (std::same_as<RightCard, stage::Exact> ||
              detail::is_bounded_stage<RightCard>) &&
             detail::IntegerValue<T>);
  [[nodiscard]] StageRef
  gather(const StageRef<std::uint32_t, stage::Exact> &indices) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != indices.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{value_, indices.value_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::Gather, inputs, detail::type<T>(),
                detail::flow_value_count(state_, indices.value_), {})};
  }
  template <class Count>
  [[nodiscard]] StageRef<T, stage::Bounded<Count>>
  gather(const StageRef<std::uint32_t, stage::Bounded<Count>> &indices) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != indices.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{value_, indices.value_, indices.count_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::Gather, inputs, detail::type<T>(),
                detail::flow_value_count(state_, indices.value_), {}),
            indices.count_};
  }
  [[nodiscard]] StageRef
  scatter(const StageRef<std::uint32_t, stage::Exact> &indices,
          Scatter options = {}) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != indices.state_) {
      detail::flow_pick(state_, 0u);
    }
    if (options.count == 0u) {
      options.count = detail::flow_value_count(state_, value_);
    }
    const std::array inputs{value_, indices.value_};
    return {state_,
            detail::flow_binary_values(state_, detail::Primitive::Scatter,
                                       inputs, detail::type<T>(), options.count,
                                       {.first = options.count})};
  }
  [[nodiscard]] StageRef<T, stage::Exact>
  scatter_reduce(const StageRef<std::uint32_t, stage::Exact> &indices,
                 const std::size_t output_count,
                 const Reduce operation = Reduce::Sum) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != indices.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{value_, indices.value_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::ScatterReduce, inputs,
                detail::type<T>(), output_count,
                {.first = output_count,
                 .mode = static_cast<std::uint32_t>(operation)})};
  }
  template <class Count>
  [[nodiscard]] StageRef<T, stage::Exact> scatter_reduce(
      const StageRef<std::uint32_t, stage::Bounded<Count>> &indices,
      const std::size_t output_count,
      const Reduce operation = Reduce::Sum) const
    requires std::same_as<Card, stage::Bounded<Count>>
  {
    if (state_ != indices.state_ || count_ != indices.count_ ||
        detail::flow_value_count(state_, value_) !=
            detail::flow_value_count(state_, indices.value_)) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{value_, indices.value_, count_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::ScatterReduce, inputs,
                detail::type<T>(), output_count,
                {.first = output_count,
                 .mode = static_cast<std::uint32_t>(operation)})};
  }
  [[nodiscard]] StageRef
  partition(const StageRef<std::uint32_t, stage::Exact> &flags) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != flags.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{flags.value_, value_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::Partition, inputs, detail::type<T>(),
                detail::flow_value_count(state_, value_), {})};
  }
  [[nodiscard]] StageRef
  segmented_scan(const StageRef<std::uint32_t, stage::Exact> &heads,
                 const Scan operation) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != heads.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{value_, heads.value_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::SegmentedScan, inputs,
                detail::type<T>(), detail::flow_value_count(state_, value_),
                {.mode = static_cast<std::uint32_t>(operation)})};
  }
  [[nodiscard]] StageRef
  segmented_reduce(const StageRef<std::uint32_t, stage::Exact> &heads,
                   const Reduce operation = Reduce::Sum) const
    requires std::same_as<Card, stage::Exact>
  {
    if (state_ != heads.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::array inputs{value_, heads.value_};
    return {state_,
            detail::flow_binary_values(
                state_, detail::Primitive::SegmentedReduce, inputs,
                detail::type<T>(), detail::flow_value_count(state_, value_),
                {.mode = static_cast<std::uint32_t>(operation)})};
  }
  [[nodiscard]] StageRef<std::uint32_t, stage::Bounded<std::uint32_t>>
  compact(Compact options = {}) const
    requires(std::same_as<Card, stage::Exact> && std::same_as<T, std::uint32_t>)
  {
    if (options.capacity == 0u) {
      options.capacity = detail::flow_value_count(state_, value_);
    }
    const detail::BoundedIds result =
        detail::flow_compact_value(state_, value_, options.capacity);
    return {state_, result.values, result.count};
  }
  [[nodiscard]] StageRef histogram(const Histogram options) const
    requires(std::same_as<Card, stage::Exact> && std::same_as<T, std::uint32_t>)
  {
    return {state_, detail::flow_unary_value(state_, value_,
                                             detail::Primitive::Histogram,
                                             detail::type<T>(), options.bins,
                                             {.first = options.bins})};
  }
  [[nodiscard]] StageRef sort() const
    requires std::same_as<Card, stage::Exact>
  {
    return {state_,
            detail::flow_unary_value(
                state_, value_, detail::Primitive::Sort, detail::type<T>(),
                detail::flow_value_count(state_, value_), {})};
  }
  [[nodiscard]] StageRef sort() const
    requires detail::is_bounded_stage<Card>
  {
    return {state_,
            detail::flow_bounded_sort_value(state_, value_, count_, false),
            count_};
  }
  [[nodiscard]] StageRef<std::uint32_t, Card> argsort() const
    requires std::same_as<Card, stage::Exact>
  {
    return {state_,
            detail::flow_unary_value(
                state_, value_, detail::Primitive::Argsort, detail::Type::U32,
                detail::flow_value_count(state_, value_), {})};
  }
  [[nodiscard]] StageRef<std::uint32_t, Card> argsort() const
    requires detail::is_bounded_stage<Card>
  {
    return {state_,
            detail::flow_bounded_sort_value(state_, value_, count_, true),
            count_};
  }
  [[nodiscard]] StageRef<T, stage::Scalar>
  reduce(const Reduce operation = Reduce::Sum) const
    requires std::same_as<Card, stage::Exact>
  {
    if (detail::flow_value_count(state_, value_) == 0u &&
        operation != Reduce::Sum) {
      detail::flow_reject(state_, Reason::ReduceCountZero);
    }
    return {state_,
            detail::flow_unary_value(
                state_, value_, detail::Primitive::Reduce, detail::type<T>(),
                1u, {.mode = static_cast<std::uint32_t>(operation)})};
  }
  [[nodiscard]] StageRef<T, stage::Scalar>
  reduce(const Reduce operation = Reduce::Sum) const
    requires detail::is_bounded_stage<Card>
  {
    return {state_, detail::flow_bounded_reduce_value(state_, value_, count_,
                                                      operation)};
  }
  [[nodiscard]] StageRef<CountFor<T>, stage::Scalar> count() const
    requires std::same_as<Card, stage::Exact>
  {
    return {state_, detail::flow_unary_value(
                        state_, value_, detail::Primitive::Reduce,
                        detail::type<CountFor<T>>(), 1u, {.flag = true})};
  }
  [[nodiscard]] StageRef<T, stage::Scalar> scalar() const
    requires std::same_as<Card, stage::Exact>
  {
    if (detail::flow_value_count(state_, value_) != 1u) {
      detail::flow_reject(state_, Reason::ScalarCountInvalid);
    }
    return {state_, value_};
  }
  [[nodiscard]] StageRef<detail::ResidentCountT<T, Card>, stage::Scalar>
  count() const
    requires detail::is_bounded_stage<Card>
  {
    return {state_, count_};
  }
#endif
