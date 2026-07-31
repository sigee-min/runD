#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
private:
  template <class Expression, class Count>
  friend auto detail::bounded_emit(const std::shared_ptr<detail::FlowState> &,
                                   std::span<const std::uint32_t>,
                                   std::string_view, const Expression &,
                                   std::uint32_t);
  template <class Expression, class Count>
  friend auto
  detail::bounded_emit_reject(const std::shared_ptr<detail::FlowState> &);
  friend struct detail::StageRefAccess;
  template <class, class, class> friend class Flow;
  template <class, class> friend class StageRef;
  template <class, class, class> friend class Groups;
  template <class, class> friend class GroupValuesRef;
  template <class> friend struct detail::SchemaRef;
  template <class...> friend class ZipRef;
  StageRef(std::shared_ptr<detail::FlowState> state, const std::uint32_t value,
           const std::uint32_t count = 0u, const MatrixShape shape = {})
      : state_(std::move(state)), value_(value), count_(count), shape_(shape) {}
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t value_{};
  std::uint32_t count_{};
  MatrixShape shape_{};
#endif
