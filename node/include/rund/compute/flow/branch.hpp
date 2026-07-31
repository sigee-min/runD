#pragma once
#include <rund/compute/flow/zip.hpp>

namespace rund::compute {
template <class... Schema> class RecordRef final {
public:
  static constexpr std::size_t size = sizeof...(Schema);

  RecordRef(const RecordRef &) = default;
  RecordRef(RecordRef &&) noexcept = default;
  RecordRef &operator=(const RecordRef &) = default;
  RecordRef &operator=(RecordRef &&) noexcept = default;
  template <std::size_t I> [[nodiscard]] auto get() const {
    using Field = std::tuple_element_t<I, std::tuple<Schema...>>;
    return detail::SchemaRef<Field>::make(state_, values_,
                                          detail::schema_offset<I, Schema...>);
  }
  template <class Tag> [[nodiscard]] auto get() const {
    constexpr std::size_t index = detail::field_index<Tag, Schema...>;
    static_assert(index < sizeof...(Schema),
                  "compute record tag must identify exactly one field");
    return get<index>();
  }

private:
  template <class Expression, class Count>
  friend auto detail::bounded_emit(const std::shared_ptr<detail::FlowState> &,
                                   std::span<const std::uint32_t>,
                                   std::string_view, const Expression &,
                                   std::uint32_t);
  template <class Expression, class Count>
  friend auto
  detail::bounded_emit_reject(const std::shared_ptr<detail::FlowState> &);
  template <class... Node>
    requires(detail::SelectionNode<Node> && ...)
  friend auto record(const Node &...values);
  template <class... Node>
    requires(detail::SelectionNode<Node> && ...)
  friend auto outputs(const Node &...values);
  template <class...> friend class RecordRef;
  template <class, class> friend class StageRef;
  template <class, class, class> friend class Flow;
  template <class...> friend class ZipRef;
  template <class...> friend class Selection;
  template <class> friend struct detail::SchemaRef;
  friend struct detail::NodeAccess;
  RecordRef(std::shared_ptr<detail::FlowState> state,
            const std::array<std::uint32_t,
                             detail::schema_leaf_count<Record<Schema...>>>
                values)
      : state_(std::move(state)), values_(values) {}
  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, detail::schema_leaf_count<Record<Schema...>>>
      values_{};
};

namespace detail {
struct NodeAccess final {
  template <class... Schema>
  [[nodiscard]] static const std::shared_ptr<FlowState> &
  state(const RecordRef<Schema...> &value) noexcept {
    return value.state_;
  }
  template <class... Schema>
  [[nodiscard]] static const auto &
  ids(const RecordRef<Schema...> &value) noexcept {
    return value.values_;
  }
};
template <class T, class Card>
[[nodiscard]] const std::shared_ptr<FlowState> &
node_state(const StageRef<T, Card> &value) noexcept {
  return StageRefAccess::state(value);
}
template <class... Schema>
[[nodiscard]] const std::shared_ptr<FlowState> &
node_state(const RecordRef<Schema...> &value) noexcept {
  return NodeAccess::state(value);
}
template <class Tag, class Node>
[[nodiscard]] const std::shared_ptr<FlowState> &
node_state(const FieldNode<Tag, Node> &field) noexcept {
  return node_state(field.value);
}
template <std::size_t N, class T, class Card>
void append_node_ids(std::array<std::uint32_t, N> &output, std::size_t &offset,
                     const StageRef<T, Card> &value) {
  output[offset++] = StageRefAccess::id(value);
  if constexpr (is_bounded_stage<Card>) {
    output[offset++] = StageRefAccess::count(value);
  }
}
template <std::size_t N, class... Schema>
void append_node_ids(std::array<std::uint32_t, N> &output, std::size_t &offset,
                     const RecordRef<Schema...> &value) {
  for (const std::uint32_t id : NodeAccess::ids(value)) {
    output[offset++] = id;
  }
}
template <std::size_t N, class Tag, class Node>
void append_node_ids(std::array<std::uint32_t, N> &output, std::size_t &offset,
                     const FieldNode<Tag, Node> &field) {
  append_node_ids(output, offset, field.value);
}
} // namespace detail

template <class Tag, detail::SelectionNode Node>
[[nodiscard]] auto field(const Node &value) {
  return detail::FieldNode<Tag, Node>{value};
}

template <class... Schema> class Selection final {
public:
  Selection(const Selection &) = default;
  Selection(Selection &&) noexcept = default;
  Selection &operator=(const Selection &) = default;
  Selection &operator=(Selection &&) noexcept = default;

private:
  template <class... Node>
    requires(detail::SelectionNode<Node> && ...)
  friend auto outputs(const Node &...values);
  template <class, class, class> friend class Flow;
  template <class... A> [[nodiscard]] auto bind() && {
    return Flow<Outputs<Schema...>(A...), stage::Exact, input::Deferred>{
        std::move(state_), values_};
  }
  Selection(std::shared_ptr<detail::FlowState> state,
            const std::array<std::uint32_t,
                             detail::schema_leaf_count<Outputs<Schema...>>>
                values)
      : state_(std::move(state)), values_(values) {}
  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, detail::schema_leaf_count<Outputs<Schema...>>>
      values_{};
};

template <class... Node>
  requires(detail::SelectionNode<Node> && ...)
[[nodiscard]] auto record(const Node &...values) {
  static_assert(sizeof...(Node) > 0u, "compute record must contain a field");
  using Result = RecordRef<detail::NodeSchemaT<Node>...>;
  constexpr std::size_t leaves =
      detail::schema_leaf_count<Record<detail::NodeSchemaT<Node>...>>;
  static_assert(leaves <= detail::MaxOutputs,
                "compute output capacity exceeded");
  const std::array states{detail::node_state(values)...};
  const auto state = states.front();
  for (const auto &candidate : states) {
    if (candidate != state) {
      detail::flow_pick(state, 0u);
    }
  }
  std::array<std::uint32_t, leaves> ids{};
  std::size_t offset = 0u;
  (detail::append_node_ids(ids, offset, values), ...);
  return Result{state, ids};
}

template <class... Node>
  requires(detail::SelectionNode<Node> && ...)
[[nodiscard]] auto outputs(const Node &...values) {
  static_assert(sizeof...(Node) > 0u, "compute outputs must not be empty");
  using Result = Selection<detail::NodeSchemaT<Node>...>;
  constexpr std::size_t leaves =
      detail::schema_leaf_count<Outputs<detail::NodeSchemaT<Node>...>>;
  static_assert(leaves > 1u, "use a stage directly for one output");
  static_assert(leaves <= detail::MaxOutputs,
                "compute output capacity exceeded");
  const std::array states{detail::node_state(values)...};
  const auto state = states.front();
  for (const auto &candidate : states) {
    if (candidate != state) {
      detail::flow_pick(state, 0u);
    }
  }
  std::array<std::uint32_t, leaves> ids{};
  std::size_t offset = 0u;
  (detail::append_node_ids(ids, offset, values), ...);
  return Result{state, ids};
}

template <class R, class... A, class Key, class Count>
class Flow<R(A...), stage::Grouped<Key, Count>, input::Bound> final {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

  template <class Fn> [[nodiscard]] auto aggregate(Fn &&function) && {
    auto result = group_.aggregate(std::forward<Fn>(function));
    using Result = std::remove_cvref_t<decltype(result)>;
    static_assert(detail::is_record<Result> || detail::is_selection<Result>,
                  "compute group aggregate must return record or outputs");
    state_.reset();
    if constexpr (detail::is_selection<Result>) {
      return std::move(result).template bind<A...>();
    } else {
      auto selected = outputs(result);
      return std::move(selected).template bind<A...>();
    }
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state, Groups<Key, R, Count> group)
      : state_(std::move(state)), group_(std::move(group)) {}
  std::shared_ptr<detail::FlowState> state_;
  Groups<Key, R, Count> group_;
};

template <class... R, class... A>
class Flow<Outputs<R...>(A...), stage::Exact, input::Deferred> final {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
  [[nodiscard]] auto compile_async() && {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Outputs<R...>(A...)>> compile() && {
    detail::flow_outputs(state_, values_);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Outputs<R...>(A...)>>::fail(compiled.reason());
    }
    return Result<Program<Outputs<R...>(A...)>>::success(
        Program<Outputs<R...>(A...)>{std::move(compiled).value()});
  }
  [[nodiscard]] Result<std::tuple<detail::HostValueT<R>...>> collect() && {
    detail::flow_outputs(state_, values_);
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::tuple<detail::HostValueT<R>...>>::fail(
          compiled.reason());
    }
    return detail::run_host_outputs<R...>(std::move(compiled).value(),
                                          detail::flow_bindings(recipe));
  }

private:
  template <class, class, class> friend class Flow;
  template <class...> friend class Selection;
  Flow(std::shared_ptr<detail::FlowState> state,
       const std::array<std::uint32_t, detail::schema_leaf_count<Outputs<R...>>>
           values)
      : state_(std::move(state)), values_(values) {}
  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, detail::schema_leaf_count<Outputs<R...>>> values_{};
};

} // namespace rund::compute
