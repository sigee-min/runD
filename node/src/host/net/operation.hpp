#pragma once

#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/ready/timed.hpp>

#include "ready/ticket.hpp"

#include <cstdint>
#include <span>

namespace rund::net::ready::timed::detail {

class Access final {
public:
  [[nodiscard]] static Wait Complete(Ticket result) noexcept {
    Wait operation{};
    operation.result_ = std::move(result);
    return operation;
  }

  [[nodiscard]] static Wait
  Defer(const SocketView socket, const Interest public_interest,
        const short interest, const std::int64_t timeout_ns,
        const ::rund::detail::task::StopIdentity stop = {}) noexcept {
    Wait operation{};
    operation.result_ = ready::detail::Access::make(::rund::ReasonCode::Ok,
                                                    socket, public_interest);
    operation.deferred_ = true;
    operation.socket_ = socket;
    operation.public_interest_ = public_interest;
    operation.interest_ = interest;
    operation.timeout_ns_ = timeout_ns;
    operation.stop_ = stop;
    return operation;
  }
};

} // namespace rund::net::ready::timed::detail

namespace rund::net::ready::many::detail {

struct Context final {
  std::span<const Request> requests{};
  std::span<Event> out{};
  Budget budget{};
  std::int64_t timeout_ns = 0;
  bool has_timeout = false;
  ::rund::detail::task::StopIdentity stop{};
  ::rund::net::ready::Set ready_set{};
};

class Access final {
public:
  [[nodiscard]] static Wait Complete(const Result result) noexcept {
    Wait operation{};
    operation.result_ = result;
    return operation;
  }

  [[nodiscard]] static Wait
  Defer(const std::span<const Request> requests, const std::span<Event> out,
        const Budget budget, const std::int64_t timeout_ns = 0,
        const bool has_timeout = false,
        const ::rund::detail::task::StopIdentity stop = {},
        const ::rund::net::ready::Set ready_set = {}) noexcept {
    Wait operation{};
    net::result::Access::set(operation.result_, ::rund::ReasonCode::Ok);
    operation.deferred_ = true;
    operation.requests_ = requests;
    operation.out_ = out;
    operation.budget_ = budget;
    operation.timeout_ns_ = timeout_ns;
    operation.has_timeout_ = has_timeout;
    operation.stop_ = stop;
    operation.ready_set_ = ready_set;
    return operation;
  }

  [[nodiscard]] static Wait Suspend(const std::uint64_t group_id) noexcept {
    Wait operation{};
    net::result::Access::set(operation.result_, ::rund::ReasonCode::Ok);
    operation.suspended_ = true;
    operation.group_id_ = group_id;
    return operation;
  }

  [[nodiscard]] static const Result &ResultOf(const Wait &operation) noexcept {
    return operation.result_;
  }

  [[nodiscard]] static bool Suspended(const Wait &operation) noexcept {
    return operation.suspended_;
  }

  [[nodiscard]] static Context Snapshot(const Wait &operation) noexcept {
    return Context{
        .requests = operation.requests_,
        .out = operation.out_,
        .budget = operation.budget_,
        .timeout_ns = operation.timeout_ns_,
        .has_timeout = operation.has_timeout_,
        .stop = operation.stop_,
        .ready_set = operation.ready_set_,
    };
  }

  static void Restore(Wait &operation, const Context context) noexcept {
    operation.requests_ = context.requests;
    operation.out_ = context.out;
    operation.budget_ = context.budget;
    operation.timeout_ns_ = context.timeout_ns;
    operation.has_timeout_ = context.has_timeout;
    operation.stop_ = context.stop;
    operation.ready_set_ = context.ready_set;
  }

  static void Reject(Wait &operation, const ReasonCode code) noexcept {
    net::result::Access::set(operation.result_, code);
    operation.deferred_ = false;
    operation.suspended_ = false;
  }

  [[nodiscard]] static std::span<Event> Output(const Wait &operation) noexcept {
    return operation.out_;
  }

  [[nodiscard]] static std::uint64_t Group(const Wait &operation) noexcept {
    return operation.group_id_;
  }
};

} // namespace rund::net::ready::many::detail
