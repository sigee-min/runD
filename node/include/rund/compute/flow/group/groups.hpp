#pragma once

#include <rund/compute/flow/group/values.hpp>

namespace rund::compute {

template <class Key, class T, class C> class Groups final {
public:
  using Value = T;
  using KeyType = Key;
  using Count = C;
  static_assert(std::same_as<Count, std::uint32_t>,
                "compute grouped count must be uint32_t");

  template <class Fn> [[nodiscard]] decltype(auto) aggregate(Fn &&function) {
    return std::forward<Fn>(function)(*this);
  }
  [[nodiscard]] StageRef<Key, stage::Bounded<Count>> key() const {
    return {state_, keys_, copy_count("group-key-count")};
  }
  [[nodiscard]] StageRef<Count, stage::Bounded<Count>> count() const {
    const auto expressions = detail::make_expr();
    Expr<Count> head{
        detail::flow_expression_input<Count>(state_, expressions, heads_, 0u)};
    const Expr<Count> unit = head - head + Count{1};
    const std::array unit_inputs{heads_};
    const std::uint32_t raw_units =
        detail::flow_map_value(state_, unit_inputs, "group-unit", unit.ref_);
    const std::size_t size = detail::flow_value_count(state_, values_);
    if (size == 0u) {
      return {state_, raw_units, copy_count("group-size-count")};
    }
    using SourceCount = CountFor<T>;
    const StageRef<SourceCount, stage::Exact> positions{
        state_, detail::flow_index(state_, detail::type<SourceCount>(), size)};
    const StageRef<SourceCount, stage::Scalar> source_count{state_,
                                                            source_count_};
    const auto active = positions.combine(
        "group-size-active", source_count,
        [](auto index, auto logical) { return mask(index < logical); });
    const auto units = StageRef<Count, stage::Exact>{state_, raw_units}.combine(
        "group-active-unit", active, [](auto value, auto selected) {
          return select(selected != Count{0}, value, Count{0});
        });
    const std::array reduce_inputs{units.value_, heads_};
    const std::uint32_t counts = detail::flow_binary_values(
        state_, detail::Primitive::SegmentedReduce, reduce_inputs,
        detail::type<Count>(), detail::flow_value_count(state_, heads_),
        {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
    return {state_, counts, copy_count("group-size-count")};
  }
  [[nodiscard]] GroupValuesRef<T, CountFor<T>> values() const {
    return {state_, values_, heads_, marks_, count_, source_count_};
  }

private:
  template <class, class> friend class StageRef;
  Groups(std::shared_ptr<detail::FlowState> state, const std::uint32_t keys,
         const std::uint32_t values, const std::uint32_t heads,
         const std::uint32_t marks, const std::uint32_t count,
         const std::uint32_t source_count)
      : state_(std::move(state)), keys_(keys), values_(values), heads_(heads),
        marks_(marks), count_(count), source_count_(source_count) {}
  [[nodiscard]] std::uint32_t copy_count(const std::string_view name) const {
    auto expressions = detail::make_expr();
    Expr<Count> count{
        detail::flow_expression_input<Count>(state_, expressions, count_, 0u)};
    const std::array inputs{count_};
    return detail::flow_map_value(state_, inputs, name, count.ref_);
  }
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t keys_{};
  std::uint32_t values_{};
  std::uint32_t heads_{};
  std::uint32_t marks_{};
  std::uint32_t count_{};
  std::uint32_t source_count_{};
};

} // namespace rund::compute
