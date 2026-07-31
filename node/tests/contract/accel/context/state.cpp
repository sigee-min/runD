#include <accel/api.hpp>

#include "local.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

namespace node_accel_contract::context {

State OpenState() {
  State state{};
  state.pick = rund::node::accel::PickAccel(Policy(rund::AccelApi::Metal));
  if (!state.pick.check.ok) {
    if (!PickUnavailableReasonIsPrecise(state.pick, rund::AccelApi::Metal)) {
      return state;
    }
    state.pick = rund::node::accel::PickAccel(Policy(rund::AccelApi::Vulkan));
  }
  if (!state.pick.check.ok) {
    state.unavailable_ok =
        PickUnavailableReasonIsPrecise(state.pick, rund::AccelApi::Vulkan);
    return state;
  }

  state.available = true;
  state.context = rund::node::accel::OpenAccel(state.pick);
  state.buffer = rund::node::accel::CreateBuffer(state.pick, BufferDesc());
  state.typed_desc = TypedDesc();
  state.typed = rund::node::accel::OpenAccelBuffer(state.context, state.buffer,
                                                   state.typed_desc);
  state.second = rund::node::accel::OpenAccel(state.pick);
  state.created =
      rund::node::accel::CreateAccelBuffer(state.context, state.typed_desc);
  return state;
}

} // namespace node_accel_contract::context
