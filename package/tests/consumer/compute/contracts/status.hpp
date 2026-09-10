#pragma once

#include <rund/compute.hpp>

inline constexpr rund::compute::Status ComputeStatusSuccess =
    rund::compute::Status::success();
inline constexpr rund::compute::Status ComputeStatusBindingFailure =
    rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
inline constexpr rund::compute::Status ComputeStatusDeviceBusy =
    rund::compute::Status::fail(rund::compute::Reason::DeviceBusy);
inline constexpr rund::compute::Status ComputeProfileUnavailable =
    rund::compute::Status::fail(rund::compute::Reason::ProfileUnavailable);
inline constexpr rund::compute::Status ComputeStatusInvalidReason =
    rund::compute::Status::fail(static_cast<rund::compute::Reason>(0xffffu));
