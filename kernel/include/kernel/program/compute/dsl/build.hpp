#pragma once

#include <kernel/program/compute/dsl/expression/context.hpp>

#include <string_view>
#include <vector>

namespace rund::compute_dsl::detail {

[[nodiscard]] bool
HasFloatingPointParam(const std::vector<BindingRuntime> &bindings) noexcept;

[[nodiscard]] rund::kernel::ComputeMap
BuildMapModel(const rund::kernel::ComputeIR &ir,
              const std::vector<BindingRuntime> &bindings) noexcept;

[[nodiscard]] rund::kernel::ComputeIR
BuildCanonical(std::string_view name, bool body_ok, const char *body_reason,
               ScalarMode mode, rund::kernel::ComputeFixedFormat format,
               const std::vector<BindingRuntime> &bindings,
               const BuildContext &context);

template <typename Body>
[[nodiscard]] rund::kernel::ComputeMap
BuildMap(const rund::kernel::ComputeIR &ir, const Body &body) noexcept {
  return BuildMapModel(ir, body.bindings());
}

template <typename Body>
[[nodiscard]] rund::kernel::ComputeIR BuildIr(const std::string_view name,
                                              const Body &body,
                                              const BuildContext &context) {
  return BuildCanonical(name, body.ok(), body.reason(), Body::scalar_mode(),
                        body.fixed_format(), body.bindings(), context);
}

} // namespace rund::compute_dsl::detail
