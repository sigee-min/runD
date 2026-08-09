#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
public:
[[nodiscard]] StageRef window(const WindowSpec options) const
  requires(std::same_as<Card, stage::Exact> || detail::is_bounded_stage<Card>)
{
  if (options.op != Window::Sum && options.op != Window::Min &&
      options.op != Window::Max) {
    detail::flow_reject(state_, Reason::WindowOpUnsupported);
    return *this;
  }
  if (options.edge != WindowEdge::Clamp && options.edge != WindowEdge::Clip) {
    detail::flow_reject(state_, Reason::WindowEdgeUnsupported);
    return *this;
  }
  if (options.radius == 0u) {
    detail::flow_reject(state_, Reason::WindowRadiusInvalid);
    return *this;
  }
  if (options.radius > (std::numeric_limits<std::size_t>::max() - 1u) / 2u) {
    detail::flow_reject(state_, Reason::WorksetOverflow);
    return *this;
  }
  const std::size_t size = detail::flow_value_count(state_, value_);
  if (size == 0u) {
    if (options.op != Window::Sum) {
      detail::flow_reject(state_, Reason::WindowCountZero);
      return *this;
    }
    return {state_,
            detail::flow_unary_value(
                state_, value_, detail::Primitive::Window, detail::type<T>(),
                0u,
                {.first = options.radius * 2u + 1u,
                 .second = 1u,
                 .third = options.radius,
                 .mode = static_cast<std::uint32_t>(options.op),
                 .extra = static_cast<std::uint32_t>(options.edge)}),
            count_};
  }
  if (options.radius > size) {
    detail::flow_reject(state_, Reason::WindowRadiusInvalid);
    return *this;
  }
  if (detail::is_bounded_stage<Card> &&
      size > std::numeric_limits<std::uint32_t>::max()) {
    detail::flow_reject(state_, Reason::WindowCountOverflow);
    return *this;
  }
  if constexpr (std::same_as<Card, stage::Exact>) {
    return {state_,
            detail::flow_unary_value(
                state_, value_, detail::Primitive::Window, detail::type<T>(),
                size,
                {.first = options.radius * 2u + 1u,
                 .second = 1u,
                 .third = options.radius,
                 .fourth = size,
                 .mode = static_cast<std::uint32_t>(options.op),
                 .extra = static_cast<std::uint32_t>(options.edge)}),
            count_};
  } else {
    const std::uint32_t output = detail::flow_bounded_window_value(
        state_, value_, count_, detail::type<T>(), size,
        {.first = options.radius * 2u + 1u,
         .second = 1u,
         .third = options.radius,
         .fourth = size,
         .mode = static_cast<std::uint32_t>(options.op),
         .extra = static_cast<std::uint32_t>(options.edge)});
    return {state_, output, count_};
  }
}
#endif
