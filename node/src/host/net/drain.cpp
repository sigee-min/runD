#include <rund/net/drain.hpp>

#include "../../runtime/platform/net.hpp"
#include "buffer.hpp"
#include "bytes.hpp"
#include "ready/ticket.hpp"

#include <algorithm>

namespace rund::net::drain::detail {

ReadResult read(ready::Ticket &&ticket, const std::span<std::byte> buffer,
                const Budget budget, const void *const state,
                const ReadHandler handler) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return fail_read(claim.code);
  }
  if (budget.max_operations == 0u) {
    ReadResult result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return result;
  }
  if (handler == nullptr || InvalidBuffer(buffer.data(), buffer.size())) {
    return fail_read(::rund::ReasonCode::TaskInvalid);
  }
  ReadResult result{::rund::ReasonCode::Ok};
  for (std::uint32_t attempt = 0u; attempt < budget.max_operations; ++attempt) {
    const ReceiveResult received =
        ::rund::net::detail::receive_attempt(claim, buffer);
    if (!received) {
      if (received.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
        return result;
      }
      net::result::Access::set(result, received.code());
      return result;
    }
    const std::size_t completed =
        received.bytes <= 0
            ? 0u
            : std::min(static_cast<std::size_t>(received.bytes), buffer.size());
    ++result.reads;
    result.bytes += completed;
    if (!handler(state, std::span<const std::byte>{buffer.data(), completed})) {
      result.handler_stopped = true;
      return result;
    }
    if (completed == 0u) {
      return result;
    }
  }
  result.budget_exhausted = true;
  return result;
}

WriteResult write(ready::Ticket &&ticket, std::span<const std::byte> bytes,
                  const Budget budget, const void *const state,
                  const WriteHandler handler) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return fail_write(claim.code);
  }
  if (bytes.empty()) {
    WriteResult result{::rund::ReasonCode::Ok};
    result.all_written = true;
    return result;
  }
  if (budget.max_operations == 0u) {
    WriteResult result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return result;
  }
  if (handler == nullptr || InvalidBuffer(bytes.data(), bytes.size())) {
    return fail_write(::rund::ReasonCode::TaskInvalid);
  }
  WriteResult result{::rund::ReasonCode::Ok};
  std::uint64_t completed_offset = 0u;
  for (std::uint32_t attempt = 0u; attempt < budget.max_operations; ++attempt) {
    const SendResult sent = ::rund::net::detail::send_attempt(claim, bytes);
    if (!sent) {
      if (sent.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
        return result;
      }
      net::result::Access::set(result, sent.code());
      return result;
    }
    const std::size_t completed =
        sent.bytes <= 0
            ? 0u
            : std::min(static_cast<std::size_t>(sent.bytes), bytes.size());
    ++result.writes;
    result.bytes += completed;
    completed_offset += static_cast<std::uint64_t>(completed);
    bytes = bytes.subspan(completed);
    const bool keep_writing = handler(state, completed_offset, sent);
    if (bytes.empty()) {
      result.all_written = true;
      result.handler_stopped = !keep_writing;
      return result;
    }
    if (!keep_writing || completed == 0u) {
      result.handler_stopped = !keep_writing;
      return result;
    }
  }
  result.budget_exhausted = true;
  return result;
}

} // namespace rund::net::drain::detail
