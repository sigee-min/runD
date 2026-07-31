#pragma once

#include <accel/buffer.hpp>

#include <node/accel/context.hpp>

#include "../local.hpp"

namespace node_accel_contract::context::reject {

struct TransferFixture {
  std::array<std::uint32_t, 8u> upload{1u, 1u, 2u, 3u, 5u, 8u, 13u, 21u};
  std::array<std::uint32_t, 8u> download{};
};

[[nodiscard]] inline bool
OpenRejects(const State &state, const rund::Buffer &buffer,
            const char *reason = "accel_context_buffer_invalid") {
  return CheckReason(rund::node::accel::OpenAccelBuffer(state.context, buffer,
                                                        state.typed_desc)
                         .check,
                     reason);
}

} // namespace node_accel_contract::context::reject
