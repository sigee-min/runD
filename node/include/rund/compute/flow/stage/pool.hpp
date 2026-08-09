#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
public:
[[nodiscard]] StageRef pool(const PoolSpec options) const
  requires(std::same_as<Card, stage::Exact>)
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
  if (options.tail != PoolTail::Drop && options.tail != PoolTail::Keep) {
    detail::flow_reject(state_, Reason::GraphPrimitiveInvalid);
    return *this;
  }
  if (options.width == 0u || options.stride == 0u) {
    detail::flow_reject(state_, Reason::WindowZero);
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
                {.first = options.width,
                 .second = options.stride,
                 .mode = static_cast<std::uint32_t>(options.op),
                 .extra = static_cast<std::uint32_t>(options.edge)}),
            count_};
  }
  if (options.width > size) {
    detail::flow_reject(state_, Reason::GraphShapeMismatch);
    return *this;
  }
  const std::size_t count = options.tail == PoolTail::Drop
                                ? 1u + (size - options.width) / options.stride
                                : 1u + (size - 1u) / options.stride;
  return {state_,
          detail::flow_unary_value(
              state_, value_, detail::Primitive::Window, detail::type<T>(),
              count,
              {.first = options.width,
               .second = options.stride,
               .fourth = count,
               .mode = static_cast<std::uint32_t>(options.op),
               .extra = static_cast<std::uint32_t>(options.edge)}),
          count_};
}
#endif
