#pragma once

#include "model.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] const char *ResidentDescReason(const ResidentDesc &desc) noexcept;

[[nodiscard]] bool ResidentRefFits(const ResidentEntry &entry,
                                   const std::shared_ptr<void> &owner,
                                   const rund::kernel::ResidentBufferRef &ref,
                                   const std::shared_ptr<void> &handle,
                                   const char *id_reason, const char *&reason,
                                   bool allow_stride = false) noexcept;

} // namespace rund::node::accel::detail
