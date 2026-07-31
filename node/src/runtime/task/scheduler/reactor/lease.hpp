#pragma once

#include "../../../../host/net/registry/socket.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace rund::node {

struct ReactorLeaseSource final {
  enum class Kind : std::uint8_t {
    Invalid,
    HostFd,
    Socket,
  };

  [[nodiscard]] static constexpr ReactorLeaseSource host() noexcept {
    return ReactorLeaseSource{.kind = Kind::HostFd};
  }

  [[nodiscard]] static constexpr ReactorLeaseSource
  socket(const ::rund::net::SocketView value) noexcept {
    return ReactorLeaseSource{.socket_view = value, .kind = Kind::Socket};
  }

  ::rund::net::SocketView socket_view{};
  Kind kind = Kind::Invalid;
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
      if (source.kind == ReactorLeaseSource::Kind::HostFd) {
        continue;
      }
      if (source.kind != ReactorLeaseSource::Kind::Socket ||
          !source.socket_view) {
        storage_.clear();
        return false;
      }
      ::rund::net::SocketLease lease =
          ::rund::net::LeaseSocket(source.socket_view);
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
