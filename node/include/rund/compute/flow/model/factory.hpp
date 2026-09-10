#pragma once

#include <rund/compute/flow/model/schema.hpp>

namespace rund::compute {
namespace detail {

template <class Stage> class StagePipe {
public:
  template <class Fn>
    requires requires(Stage &&stage, Fn &&function) {
      static_cast<Stage &&>(stage).pipe_stage(std::forward<Fn>(function));
    }
  [[nodiscard]] decltype(auto) pipe(Fn &&function) && {
    return static_cast<Stage &&>(*this).pipe_stage(std::forward<Fn>(function));
  }

protected:
  StagePipe() = default;
};
struct FlowFactory final {
  template <ComputeValue T>
  [[nodiscard]] static Flow<T(T), stage::Exact> make(Target target,
                                                     std::span<const T> input);
};
} // namespace detail

template <class Fn, detail::ComputeValue... Values>
[[nodiscard]] auto capture(Fn &&function, Values... values) {
  using Function = std::remove_cvref_t<Fn>;
  static_assert(std::is_empty_v<Function>,
                "compute::capture requires a captureless element function");
  static_assert(std::is_trivially_copy_constructible_v<Function>,
                "compute::capture requires a trivial element function");
  return detail::CapturedElement<Function, Values...>{
      std::forward<Fn>(function), std::tuple<Values...>{std::move(values)...}};
}

} // namespace rund::compute
