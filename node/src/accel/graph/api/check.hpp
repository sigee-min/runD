#pragma once

#include <accel/kernel/check.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "../../context/internal.hpp"

#include <kernel/program/compute/model.hpp>

#include <string_view>

namespace rund::node::accel {

[[nodiscard]] inline rund::AccelKernelCheck
RejectGraphCheck(const char *const reason) noexcept {
  return rund::AccelKernelCheck{false, reason};
}

[[nodiscard]] inline rund::AccelKernelCheck OkGraphCheck() noexcept {
  return rund::AccelKernelCheck{true, "ok"};
}

[[nodiscard]] inline bool ReasonOk(const char *const reason) noexcept {
  return std::string_view{reason == nullptr ? "" : reason} ==
         std::string_view{"ok"};
}

[[nodiscard]] inline rund::AccelKernel
RejectKernel(const char *const reason,
             const detail::ContextAdmission &admission = {}) {
  return rund::AccelKernel{
      .check = RejectGraphCheck(reason),
      .api = admission.api,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .frozen_caps = admission.caps,
      .context_id = admission.context_id,
      .reason = reason,
  };
}

} // namespace rund::node::accel
