#include <rund/net/frame/io.hpp>

#include <rund/net/frame/length.hpp>
#include <rund/net/vectored.hpp>

#include "../../../runtime/platform/net.hpp"
#include "../../../runtime/platform/net/vectored.hpp"
#include "../../../runtime/task/scheduler/access.hpp"
#include "../buffer.hpp"
#include "../bytes.hpp"
#include "../ready/ticket.hpp"
#include "../vectored.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace rund::net::frame {
namespace {

[[nodiscard]] WriteResult fail_write(const ::rund::ReasonCode code) noexcept {
  return WriteResult{code};
}

[[nodiscard]] ReadResult fail_read(const ::rund::ReasonCode code) noexcept {
  return ReadResult{code};
}

struct WriteBatch final {
  std::array<batch::Slice, 2u> slices{};
  node::nativeio::VectoredBatch native{.valid = true};

  void add(const std::span<const std::byte> bytes) noexcept {
    const std::size_t index = native.count++;
    slices[index] = batch::Slice{.data = bytes.data(), .size = bytes.size()};
    native.native[index] = node::nativeio::NativeSlice{
        .iov_base = const_cast<void *>(static_cast<const void *>(bytes.data())),
        .iov_len = bytes.size(),
    };
  }

  void begin(const std::uint64_t bytes) noexcept {
    native.count = 0u;
    native.admitted_bytes = bytes;
  }

  [[nodiscard]] std::span<const batch::Slice> view() const noexcept {
    return std::span<const batch::Slice>{slices}.first(native.count);
  }
};

class WritePlan final {
public:
  WritePlan(const std::span<const std::byte> header,
            const std::span<const std::byte> payload) noexcept
      : header_(header), payload_(payload),
        bytes_(static_cast<std::uint64_t>(header.size()) +
               static_cast<std::uint64_t>(payload.size())) {}

  void remaining(WriteBatch &batch, const std::size_t header_bytes,
                 const std::size_t payload_bytes) const noexcept {
    const std::span<const std::byte> header = header_.subspan(header_bytes);
    const std::span<const std::byte> payload = payload_.subspan(payload_bytes);

    batch.begin(bytes_ - static_cast<std::uint64_t>(header_bytes) -
                static_cast<std::uint64_t>(payload_bytes));
    if (!header.empty()) {
      batch.add(header);
    }
    if (!payload.empty()) {
      batch.add(payload);
    }
  }

private:
  std::span<const std::byte> header_{};
  std::span<const std::byte> payload_{};
  std::uint64_t bytes_ = 0u;
};

[[nodiscard]] ::rund::ReasonCode
write_shape(const std::span<const std::byte> payload) noexcept {
  const std::size_t capacity = std::min<std::size_t>(
      node::scheduler_access::ActiveLimits().net_iov_capacity,
      node::nativeio::VectoredCapacity);
  const std::size_t slice_count = payload.empty() ? 1u : 2u;
  constexpr std::size_t header_size = 4u;
  constexpr std::size_t byte_capacity =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (slice_count > capacity || InvalidBuffer(payload.data(), payload.size()) ||
      header_size > byte_capacity ||
      payload.size() > byte_capacity - header_size) {
    return ::rund::ReasonCode::TaskInvalid;
  }
  return ::rund::ReasonCode::Ok;
}

constexpr void advance(WriteResult &result, std::size_t completed,
                       const std::size_t payload_size) noexcept {
  const std::size_t header_remaining =
      4u - static_cast<std::size_t>(result.header_bytes);
  const std::size_t header_bytes = std::min(completed, header_remaining);
  result.header_bytes += static_cast<std::uint32_t>(header_bytes);
  completed -= header_bytes;

  const std::size_t payload_remaining =
      payload_size - static_cast<std::size_t>(result.payload_bytes);
  const std::size_t payload_bytes = std::min(completed, payload_remaining);
  result.payload_bytes += static_cast<std::uint32_t>(payload_bytes);
  result.bytes = result.payload_bytes;
  result.header_written = result.header_bytes == 4u;
  result.payload_written =
      result.header_written && result.payload_bytes == payload_size;
}

} // namespace

WriteResult write(ready::Ticket &&ticket,
                  const std::span<const std::byte> payload,
                  const IoLimit limit) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Writable);
  if (!claim) {
    return fail_write(claim.code);
  }
  if (payload.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return fail_write(::rund::ReasonCode::NetFrameTooLarge);
  }

  std::array<std::byte, 4u> header{};
  const Result encoded =
      encode_length(static_cast<std::uint32_t>(payload.size()),
                    std::span<std::byte>{header}, limit.frame);
  if (!encoded) {
    return fail_write(encoded.code());
  }
  if (limit.max_writes == 0u) {
    WriteResult result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return result;
  }

  ready::detail::Operation operation = ready::detail::prepare(claim);
  if (!operation) {
    return fail_write(operation.code());
  }
  const ::rund::ReasonCode shape = write_shape(payload);
  if (shape != ::rund::ReasonCode::Ok) {
    return fail_write(shape);
  }
  const WritePlan plan{header, payload};

  WriteResult result{::rund::ReasonCode::Ok};
  WriteBatch batch{};
  std::uint32_t writes = 0u;
  while (!result.complete()) {
    if (writes >= limit.max_writes) {
      result.budget_exhausted = true;
      return result;
    }
    ++writes;

    plan.remaining(batch, result.header_bytes, result.payload_bytes);
    const std::uint64_t admitted = batch.native.admitted_bytes;
    const node::NativeCallResult native =
        node::NativeSendVectored(operation.native(), batch.native);
    const SendResult sent = batch::detail::complete_send(
        operation.id(), batch.view(), admitted, native);
    if (!sent) {
      if (sent.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
        return result;
      }
      net::result::Access::set(result, sent.code());
      return result;
    }
    const std::size_t completed =
        sent.bytes <= 0 ? 0u
                        : std::min(static_cast<std::size_t>(sent.bytes),
                                   static_cast<std::size_t>(admitted));
    advance(result, completed, payload.size());
    if (completed == 0u && !result.complete()) {
      return result;
    }
  }
  return result;
}

ReadResult read(ready::Ticket &&ticket, const std::span<std::byte> payload,
                const IoLimit limit) noexcept {
  const ready::detail::Claim claim =
      ready::detail::claim(std::move(ticket), ready::Interest::Readable);
  if (!claim) {
    return fail_read(claim.code);
  }
  if (limit.max_reads == 0u) {
    ReadResult result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return result;
  }
  ready::detail::Operation operation = ready::detail::prepare(claim);
  if (!operation) {
    return fail_read(operation.code());
  }

  std::array<std::byte, 4u> header{};
  ReadResult result{::rund::ReasonCode::Ok};
  std::uint32_t reads = 0u;
  std::size_t header_offset = 0u;
  std::size_t payload_offset = 0u;

  while (header_offset < header.size()) {
    if (reads >= limit.max_reads) {
      result.budget_exhausted = true;
      return result;
    }
    ++reads;
    const std::span<std::byte> remaining =
        std::span<std::byte>{header}.subspan(header_offset);
    const ReceiveResult received = net::detail::complete_receive(
        operation.id(), remaining,
        node::NativeTryRecv(operation.native(), remaining));
    if (!received) {
      if (received.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
        return result;
      }
      net::result::Access::set(result, received.code());
      return result;
    }
    const std::uint32_t completed =
        received.bytes <= 0
            ? 0u
            : static_cast<std::uint32_t>(std::min<std::size_t>(
                  static_cast<std::size_t>(received.bytes), remaining.size()));
    header_offset += completed;
    result.header_bytes = static_cast<std::uint32_t>(header_offset);
    result.header_read = header_offset == header.size();
    if (completed == 0u) {
      return result;
    }
  }

  const Result frame =
      decode_length(std::span<const std::byte>{header}, limit.frame);
  if (!frame) {
    net::result::Access::set(result, frame.code());
    result.bytes = 0u;
    return result;
  }
  if (frame.bytes > payload.size()) {
    net::result::Access::set(result,
                             ::rund::ReasonCode::NetFrameBufferTooSmall);
    result.bytes = 0u;
    return result;
  }
  if (frame.bytes == 0u) {
    result.payload_read = true;
    return result;
  }

  while (payload_offset < frame.bytes) {
    if (reads >= limit.max_reads) {
      result.budget_exhausted = true;
      return result;
    }
    ++reads;
    const std::span<std::byte> remaining =
        payload.subspan(payload_offset, frame.bytes - payload_offset);
    const ReceiveResult received = net::detail::complete_receive(
        operation.id(), remaining,
        node::NativeTryRecv(operation.native(), remaining));
    if (!received) {
      if (received.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
        return result;
      }
      net::result::Access::set(result, received.code());
      return result;
    }
    const std::uint32_t completed =
        received.bytes <= 0
            ? 0u
            : static_cast<std::uint32_t>(std::min<std::size_t>(
                  static_cast<std::size_t>(received.bytes), remaining.size()));
    payload_offset += completed;
    result.payload_bytes = static_cast<std::uint32_t>(payload_offset);
    result.bytes = static_cast<std::uint32_t>(payload_offset);
    result.payload_read = payload_offset == frame.bytes;
    if (result.payload_read || completed == 0u) {
      return result;
    }
  }
  return result;
}

} // namespace rund::net::frame
