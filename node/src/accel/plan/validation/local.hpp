#pragma once

#include "../validation.hpp"
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool ReasonIsOk(const char *const reason) noexcept {
  return reason != nullptr && std::string_view{reason} == "ok";
}

[[nodiscard]] bool
BindingInputCountMatches(const rund::kernel::ComputePlan &plan,
                         const rund::kernel::BindingSet &bindings,
                         PlanBindingInputMode input_mode) noexcept;

} // namespace rund::node::accel::detail
