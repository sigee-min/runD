#pragma once

#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/options.hpp>
#include <rund/net/vectored.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node {

namespace nativeio {
struct VectoredBatch;
}

enum class NativeCallState : std::uint8_t {
  Complete,
  Failed,
  InvalidInput,
  Unsupported,
  WouldBlock,
  InProgress,
};

struct NativeCallResult {
  std::int64_t value = -1;
  int error = 0;
  NativeCallState state = NativeCallState::Failed;
};

struct NativeAddressResult {
  NativeCallResult call{};
  ::rund::net::Address address{};
};

[[nodiscard]] NativeCallResult
NativeSocket(::rund::net::Family family,
             ::rund::net::Transport transport) noexcept;
[[nodiscard]] NativeCallResult
NativeBind(int fd, const ::rund::net::Address &address) noexcept;
[[nodiscard]] NativeCallResult NativeListen(int fd, int backlog) noexcept;
[[nodiscard]] NativeAddressResult NativeGetSockName(int fd) noexcept;
[[nodiscard]] NativeCallResult
NativeShutdown(int fd, ::rund::net::ShutdownMode mode) noexcept;
[[nodiscard]] NativeCallResult NativePrepareSocketSend(int fd) noexcept;
[[nodiscard]] NativeCallResult NativeRecv(int fd,
                                          std::span<std::byte> buffer) noexcept;
[[nodiscard]] NativeCallResult
NativeSend(int fd, std::span<const std::byte> buffer) noexcept;
[[nodiscard]] NativeCallResult
NativeTryRecv(int fd, std::span<std::byte> buffer) noexcept;
[[nodiscard]] NativeCallResult
NativeTrySend(int fd, std::span<const std::byte> buffer) noexcept;
[[nodiscard]] NativeAddressResult
NativeRecvFrom(int fd, std::span<std::byte> buffer) noexcept;
[[nodiscard]] NativeCallResult
NativeSendTo(int fd, std::span<const std::byte> buffer,
             const ::rund::net::Address &address) noexcept;
[[nodiscard]] NativeCallResult
NativeRecvVectored(int fd, const nativeio::VectoredBatch &batch) noexcept;
[[nodiscard]] NativeCallResult
NativeSendVectored(int fd, const nativeio::VectoredBatch &batch) noexcept;
[[nodiscard]] NativeAddressResult NativeAccept(int fd) noexcept;
[[nodiscard]] NativeCallResult
NativeConnect(int fd, const ::rund::net::Address &address) noexcept;
[[nodiscard]] NativeCallResult
NativeGetSocketError(int fd, const ::rund::net::Address &address) noexcept;
[[nodiscard]] NativeCallResult
NativeSetSocketOption(int fd, ::rund::net::option::Name option,
                      ::rund::net::option::Value value) noexcept;
[[nodiscard]] NativeCallResult
NativeGetSocketOption(int fd, ::rund::net::option::Name option) noexcept;

} // namespace rund::node
