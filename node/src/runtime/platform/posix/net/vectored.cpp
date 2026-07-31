#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>

#include <cerrno>

#include "../../net.hpp"
#include "../../net/vectored.hpp"
#include "result.hpp"
#include "socket/call.hpp"

namespace rund::node {

NativeVectoredResult
NativeRecvVectored(const int fd,
                   const nativeio::VectoredBatch &batch) noexcept {
  msghdr message{};
  message.msg_iov = const_cast<iovec *>(batch.native.data());
  message.msg_iovlen = static_cast<decltype(message.msg_iovlen)>(batch.count);
  errno = 0;
  const ssize_t value = ::recvmsg(fd, &message, posix_net::TryRecvFlags());
  return NativeVectoredResult{
      .call = PosixNetResult(value, value < 0 ? errno : 0),
      .admitted_bytes = batch.admitted_bytes,
  };
}

NativeVectoredResult
NativeSendVectored(const int fd,
                   const nativeio::VectoredBatch &batch) noexcept {
  msghdr message{};
  message.msg_iov = const_cast<iovec *>(batch.native.data());
  message.msg_iovlen = static_cast<decltype(message.msg_iovlen)>(batch.count);
  errno = 0;
  const ssize_t value = ::sendmsg(fd, &message, posix_net::TrySendFlags());
  return NativeVectoredResult{
      .call = PosixNetResult(value, value < 0 ? errno : 0),
      .admitted_bytes = batch.admitted_bytes,
  };
}

} // namespace rund::node
