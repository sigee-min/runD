#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../internal/admission.hpp"
#include "reference.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool
ContextMatchesAdmission(const rund::AccelContext &context,
                        const ContextAdmission &admission) noexcept {
  return admission.check.ok && context.check.ok &&
         context.id == admission.context_id && context.api == admission.api &&
         SameObject(context.owner, admission.owner) &&
         SameCaps(context.caps, admission.caps) && context.pick.check.ok &&
         context.pick.api == admission.api && context.pick.owner != nullptr &&
         SameObject(context.pick.owner, admission.pick) &&
         SameCaps(context.pick.caps, admission.caps);
}

[[nodiscard]] bool
AccelBufferOwnerMatches(const ContextAdmission &admission,
                        const rund::AccelBuffer &buffer) noexcept {
  return buffer.context_id == admission.context_id &&
         SameObject(buffer.owner, admission.owner);
}

[[nodiscard]] bool BufferOwnerMatches(const ContextAdmission &admission,
                                      const rund::AccelContext &,
                                      const rund::Buffer &buffer) noexcept {
  return SameObject(admission.owner, buffer.owner) ||
         SameObject(admission.pick, buffer.owner);
}

} // namespace rund::node::accel::detail
