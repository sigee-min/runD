#pragma once

#include <rund/task/await.hpp>
#include <rund/net/frame/limit.hpp>
#include <rund/net/frame/result/read.hpp>
#include <rund/net/frame/result/write.hpp>
#include <rund/net/ready.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <utility>

namespace rund::net::frame {

[[nodiscard]] WriteResult
write(ready::Ticket &&ticket, std::span<const std::byte> payload,
      IoLimit limit = {}) noexcept;

[[nodiscard]] ReadResult read(ready::Ticket &&ticket,
                              std::span<std::byte> payload,
                              IoLimit limit = {}) noexcept;

[[nodiscard]] inline task::Task<WriteResult>
write(const SocketView socket, const std::span<const std::byte> payload,
      const IoLimit limit = {}) {
  if (payload.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    co_return WriteResult{::rund::ReasonCode::NetFrameTooLarge};
  }
  if (limit.max_writes == 0u) {
    WriteResult result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    co_return result;
  }
  ready::Ticket ticket = co_await ready::write(socket);
  co_return write(std::move(ticket), payload, limit);
}

[[nodiscard]] inline task::Task<ReadResult>
read(const SocketView socket, const std::span<std::byte> payload,
     const IoLimit limit = {}) {
  if (limit.max_reads == 0u) {
    ReadResult result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    co_return result;
  }
  ready::Ticket ticket = co_await ready::read(socket);
  co_return read(std::move(ticket), payload, limit);
}

} // namespace rund::net::frame
