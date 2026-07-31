#include "../net.hpp"
#include "../net/vectored.hpp"

#include <limits>

namespace rund::node {
namespace {

[[nodiscard]] constexpr NativeCallResult UnsupportedNet() noexcept {
  return NativeCallResult{.state = NativeCallState::Unsupported};
}

} // namespace

NativeCallResult NativeSocket(const ::rund::net::Family family,
                              const ::rund::net::Transport transport) noexcept {
  (void)family;
  (void)transport;
  return UnsupportedNet();
}

NativeCallResult NativeBind(const int fd,
                            const ::rund::net::Address &address) noexcept {
  (void)fd;
  (void)address;
  return UnsupportedNet();
}

NativeCallResult NativeListen(const int fd, const int backlog) noexcept {
  (void)fd;
  (void)backlog;
  return UnsupportedNet();
}

NativeAddressResult NativeGetSockName(const int fd) noexcept {
  (void)fd;
  return NativeAddressResult{.call = UnsupportedNet()};
}

NativeCallResult NativeShutdown(const int fd,
                                const ::rund::net::ShutdownMode mode) noexcept {
  (void)fd;
  (void)mode;
  return UnsupportedNet();
}

NativeCallResult NativePrepareSocketSend(const int fd) noexcept {
  (void)fd;
  return UnsupportedNet();
}

NativeCallResult NativeRecv(const int fd,
                            const std::span<std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return UnsupportedNet();
}

NativeCallResult NativeSend(const int fd,
                            const std::span<const std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return UnsupportedNet();
}

NativeCallResult NativeTryRecv(const int fd,
                               const std::span<std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return UnsupportedNet();
}

NativeCallResult
NativeTrySend(const int fd, const std::span<const std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return UnsupportedNet();
}

NativeAddressResult NativeRecvFrom(const int fd,
                                   const std::span<std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return NativeAddressResult{.call = UnsupportedNet()};
}

NativeCallResult NativeSendTo(const int fd,
                              const std::span<const std::byte> buffer,
                              const ::rund::net::Address &address) noexcept {
  (void)fd;
  (void)buffer;
  (void)address;
  return UnsupportedNet();
}

NativeVectoredResult
NativeRecvVectored(const int fd,
                   const nativeio::VectoredBatch &batch) noexcept {
  (void)fd;
  return NativeVectoredResult{.call = UnsupportedNet(),
                              .admitted_bytes = batch.admitted_bytes};
}

NativeVectoredResult
NativeSendVectored(const int fd,
                   const nativeio::VectoredBatch &batch) noexcept {
  (void)fd;
  return NativeVectoredResult{.call = UnsupportedNet(),
                              .admitted_bytes = batch.admitted_bytes};
}

NativeAddressResult NativeAccept(const int fd) noexcept {
  (void)fd;
  return NativeAddressResult{.call = UnsupportedNet()};
}

NativeCallResult NativeConnect(const int fd,
                               const ::rund::net::Address &address) noexcept {
  (void)fd;
  (void)address;
  return UnsupportedNet();
}

NativeCallResult
NativeGetSocketError(const int fd,
                     const ::rund::net::Address &address) noexcept {
  (void)fd;
  (void)address;
  return UnsupportedNet();
}

NativeCallResult
NativeSetSocketOption(const int fd, const ::rund::net::option::Name option,
                      const ::rund::net::option::Value value) noexcept {
  (void)fd;
  (void)option;
  (void)value;
  return UnsupportedNet();
}

NativeCallResult
NativeGetSocketOption(const int fd,
                      const ::rund::net::option::Name option) noexcept {
  (void)fd;
  (void)option;
  return UnsupportedNet();
}

} // namespace rund::node
