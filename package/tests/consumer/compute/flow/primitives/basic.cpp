#include "model.hpp"

namespace package_compute {

int Basic() {
  auto partitioned = rund::compute::on(rund::compute::Target::cpu(), values)
                         .partition(flags)
                         .collect();
  if (!partitioned) {
    return partitioned.exit_code();
  }
  if (partitioned->size() != values.size()) {
    return FlowMismatch(__LINE__);
  }

  auto windowed = rund::compute::on(rund::compute::Target::cpu(), values)
                      .window({.op = rund::compute::Window::Max, .radius = 1u})
                      .collect();
  if (!windowed) {
    return windowed.exit_code();
  }
  if (windowed->size() != values.size()) {
    return FlowMismatch(__LINE__);
  }

  auto selected =
      rund::compute::on(rund::compute::Target::cpu(), values)
          .map("select",
               [](auto value) {
                 return rund::compute::select((value > 1u) && (value < 4u),
                                              value * 2u, value + 1u);
               })
          .collect();
  if (!selected) {
    return selected.exit_code();
  }
  if (*selected != std::vector<std::uint32_t>{6u, 2u, 5u, 4u}) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
