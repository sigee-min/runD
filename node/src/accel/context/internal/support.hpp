#pragma once

#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/value.hpp>

#include "execution.hpp"

#include <node/accel/context.hpp>

#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] ContextAdmission
AdmitContextForSupport(const rund::AccelContext &context);

[[nodiscard]] KernelAdmission
AdmitKernelForSupport(const rund::AccelContext &context,
                      const rund::AccelKernel &kernel);

[[nodiscard]] KernelExecution
AdmitKernelForExecution(const rund::AccelContext &context,
                        const rund::AccelKernel &kernel);

[[nodiscard]] rund::AccelCheck
ValidateAccelBufferForSupport(const rund::AccelContext &context,
                              const rund::AccelBuffer &buffer);

[[nodiscard]] rund::AccelCheck
ValidateAccelBufferForSupport(const rund::AccelContext &context,
                              const ContextAdmission &admission,
                              const rund::AccelBuffer &buffer);

[[nodiscard]] rund::AccelCheck
ValidateAccelBufferForSupport(const ContextAdmission &admission,
                              const rund::AccelBuffer &buffer);

[[nodiscard]] rund::AccelCheck
ValidateAccelBufferForSupport(const ContextAdmission &admission,
                              const rund::AccelBuffer &buffer,
                              std::shared_ptr<void> &backend_handle);

} // namespace rund::node::accel::detail
