#pragma once

#include "../../interface/api.hpp"

#include "../../evidence.hpp"
#include "../../model.hpp"

#include "../../../evidence.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {
namespace prepared_residency {

[[nodiscard]] bool SameCheck(const rund::AccelCheck left,
                             const rund::AccelCheck right) noexcept;

[[nodiscard]] bool
SameWindowReceipt(const BackendResidencyWindowReceipt &left,
                  const BackendResidencyWindowReceipt &right) noexcept;

[[nodiscard]] std::size_t WindowStates(
    const PreparedResidencyWindowRequest &request,
    std::array<prepared::PipelineState *, ResidencyWindowCapacity> &states,
    const rund::AccelContext &context, const BackendOps *&ops) noexcept;

void CompleteResidencyWindowRelease(
    void *raw, BackendResidencyWindowRelease &&release) noexcept;

void CompleteResidencyWindowFinal(
    void *raw, BackendResidencyWindowFinal &&final) noexcept;

} // namespace prepared_residency
} // namespace rund::node::accel::detail
