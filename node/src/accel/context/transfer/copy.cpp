#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../../backend/resource.hpp"
#include "../../kernel/prepared/interface/api.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel {

detail::AccelCopy
detail::CopyAccelBuffers(const rund::AccelContext &context,
                         const std::span<const CopyEntry> requests,
                         const std::span<CopyRoute> routes,
                         const TransferAuthority authority) {
  if (requests.empty() || routes.size() < requests.size()) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const std::shared_ptr<ContextToken> context_token =
      AdmitContextToken(context);
  if (context_token == nullptr) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const std::shared_ptr<PickToken> &pick = context_token->pick;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const CopyEntry &request = requests[index];
    if (request.source == nullptr || request.target == nullptr) {
      return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
    }
    const TransferAdmission source =
        authority == TransferAuthority::PipelinePrivate
            ? AdmitAccelBufferPrivateTransfer(context_token, *request.source)
            : AdmitAccelBufferTransfer(context_token, *request.source);
    const TransferAdmission target =
        authority == TransferAuthority::PipelinePrivate
            ? AdmitAccelBufferPrivateTransfer(context_token, *request.target)
            : AdmitAccelBufferTransfer(context_token, *request.target);
    if (!source.check.ok) {
      return {.check = source.check};
    }
    if (!target.check.ok) {
      return {.check = target.check};
    }
    if (!rund::kernel::checked::add(request.source_offset, request.bytes) ||
        request.source_offset + request.bytes > source.byte_extent ||
        !rund::kernel::checked::add(request.target_offset, request.bytes) ||
        request.target_offset + request.bytes > target.byte_extent) {
      return {.check = RejectAccelCheck("accel_buffer_copy_overflow")};
    }
    routes[index] = CopyRoute{
        .source = source.route.ref,
        .source_handle = source.route.handle,
        .target = target.route.ref,
        .target_handle = target.route.handle,
        .bytes = request.bytes,
        .source_offset = request.source_offset,
        .target_offset = request.target_offset,
    };
  }
  const BackendCopy copied =
      CopyBackendBuffers(pick, routes.first(requests.size()), authority);
  return {.check =
              TransferCheckFrom(copied.check, "accel_buffer_copy_overflow"),
          .command_submits = copied.command_submits};
}

} // namespace rund::node::accel
