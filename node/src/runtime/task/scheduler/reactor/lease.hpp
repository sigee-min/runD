#pragma once

#include "../../../../host/net/registry/socket.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace rund::node {

enum class ReactorLeaseSourceDisposition : std::uint8_t {
  Invalid,
  HostFd,
  Socket,
};

class ReactorLeaseSource final {
public:
  [[nodiscard]] static constexpr ReactorLeaseSource invalid() noexcept {
    return ReactorLeaseSource{ReactorLeaseSourceDisposition::Invalid};
  }

  [[nodiscard]] static constexpr ReactorLeaseSource host_fd() noexcept {
    return ReactorLeaseSource{ReactorLeaseSourceDisposition::HostFd};
  }

  [[nodiscard]] static constexpr ReactorLeaseSource
  socket(const ::rund::net::SocketView value) noexcept {
    return value ? ReactorLeaseSource{value} : invalid();
  }

  [[nodiscard]] constexpr ReactorLeaseSourceDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr ::rund::net::SocketView socket_view() const noexcept {
    return disposition_ == ReactorLeaseSourceDisposition::Socket
               ? socket_view_
               : ::rund::net::SocketView{};
  }

private:
  explicit constexpr ReactorLeaseSource(
      const ReactorLeaseSourceDisposition disposition) noexcept
      : disposition_{disposition} {}

  explicit constexpr ReactorLeaseSource(
      const ::rund::net::SocketView socket_view) noexcept
      : socket_view_{socket_view},
        disposition_{ReactorLeaseSourceDisposition::Socket} {}

  ::rund::net::SocketView socket_view_{};
  ReactorLeaseSourceDisposition disposition_ =
      ReactorLeaseSourceDisposition::Invalid;
};

class ReactorLeaseScope final {
public:
  explicit ReactorLeaseScope(
      std::vector<::rund::net::SocketLease> &storage) noexcept
      : storage_(storage) {
    storage_.clear();
  }

  ~ReactorLeaseScope() { storage_.clear(); }

  ReactorLeaseScope(const ReactorLeaseScope &) = delete;
  ReactorLeaseScope &operator=(const ReactorLeaseScope &) = delete;

  template <typename Range, typename SourceOf>
  [[nodiscard]] bool acquire(const Range &values, SourceOf source_of) noexcept {
    if (values.size() > storage_.capacity()) {
      return false;
    }
    for (const auto &value : values) {
      const ReactorLeaseSource source = source_of(value);
      switch (source.disposition()) {
      case ReactorLeaseSourceDisposition::Invalid:
        storage_.clear();
        return false;
      case ReactorLeaseSourceDisposition::HostFd:
        continue;
      case ReactorLeaseSourceDisposition::Socket:
        break;
      }
      ::rund::net::SocketLease lease =
          ::rund::net::LeaseSocket(source.socket_view());
      if (!lease) {
        storage_.clear();
        return false;
      }
      storage_.push_back(std::move(lease));
    }
    return true;
  }

  [[nodiscard]] std::span<const ::rund::net::SocketLease>
  values() const noexcept {
    return storage_;
  }

private:
  std::vector<::rund::net::SocketLease> &storage_;
};

} // namespace rund::node
