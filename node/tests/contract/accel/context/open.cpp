#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

namespace node_accel_contract {

bool PublicContextApiOpensRealBackend() {
  namespace ctx = node_accel_contract::context;
  const ctx::State state = ctx::OpenState();
  if (!state.available) {
    return state.unavailable_ok;
  }
  if (!state.context.check.ok || state.context.api != state.pick.api ||
      !rund::node::test::SameOwner(state.context.pick.owner,
                                   state.pick.owner) ||
      state.context.id == 0u || state.context.owner == nullptr ||
      rund::node::test::SameOwner(state.context.owner, state.pick.owner) ||
      !state.context.evidence.ok ||
      std::string_view{state.context.evidence.reason} != "ok") {
    return false;
  }

  rund::AccelDevice forged_caps = state.pick;
  ++forged_caps.caps.max_window_tiles;
  if (!ctx::CheckReason(rund::node::accel::OpenAccel(forged_caps).check,
                        "accel_context_pick_invalid")) {
    return false;
  }

  rund::AccelDevice alias_owner_pick = state.pick;
  alias_owner_pick.owner =
      std::shared_ptr<void>(state.pick.owner.get(), [](void *) {});
  if (!ctx::CheckReason(rund::node::accel::OpenAccel(alias_owner_pick).check,
                        "accel_context_pick_invalid")) {
    return false;
  }

  int alias_value{};
  rund::AccelDevice shifted_owner_pick = state.pick;
  shifted_owner_pick.owner =
      std::shared_ptr<void>(state.pick.owner, &alias_value);
  if (!ctx::CheckReason(rund::node::accel::OpenAccel(shifted_owner_pick).check,
                        "accel_context_pick_invalid")) {
    return false;
  }

  if (!state.buffer.check.ok || !state.typed.check.ok ||
      std::string_view{state.typed.reason} != "ok" ||
      state.typed.byte_extent != 32u || state.typed.scalar_width_bytes != 4u ||
      state.typed.count != 8u ||
      state.typed.usage != rund::BufferUsage::ReadWrite ||
      !rund::node::test::SameOwner(state.typed.owner, state.context.owner) ||
      state.typed.handle == state.buffer.handle ||
      state.typed.buffer.handle != state.typed.handle ||
      !rund::node::test::SameOwner(state.typed.buffer.owner,
                                   state.context.owner) ||
      state.typed.resident.id != state.buffer.id ||
      state.typed.resident.bytes != state.buffer.bytes ||
      state.typed.resident.element_bytes != 4u ||
      state.typed.resident.stride_bytes != 4u ||
      state.typed.resident.count != state.typed_desc.count ||
      state.typed.resident.usage != rund::kernel::kResidentUsageWrite) {
    return false;
  }

  return state.second.check.ok && state.second.id != state.context.id &&
         !rund::node::test::SameOwner(state.second.owner, state.context.owner);
}

} // namespace node_accel_contract
