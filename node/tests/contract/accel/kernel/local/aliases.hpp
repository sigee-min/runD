#pragma once

#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include "src/accel/context/internal.hpp"

namespace node_accel_contract::kernel_case {

using CompileAccelKernelFn = rund::AccelKernel (*)(const rund::AccelContext &,
                                                   const rund::AccelGraph &);
using AdmitKernelForSupportFn = rund::node::accel::detail::KernelAdmission (*)(
    const rund::AccelContext &, const rund::AccelKernel &);
using RunAccelKernelFn = rund::AccelEvidence (*)(const rund::AccelContext &,
                                                 const rund::AccelKernel &,
                                                 const rund::AccelRun &);

} // namespace node_accel_contract::kernel_case
