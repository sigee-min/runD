#include "test/assert.hpp"

#include "net/basic/local.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/datagram.hpp>
#include <rund/net/drain.hpp>
#include <rund/net/flow/result.hpp>
#include <rund/net/frame/io.hpp>
#include <rund/net/frame/length.hpp>
#include <rund/net/handoff.hpp>
#include <rund/net/limits.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/options.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/net/server.hpp>
#include <rund/net/socket.hpp>

#include <chrono>
#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

template <typename Result>
concept ProductResult =
    std::same_as<decltype(Result{}.code()), rund::ReasonCode> &&
    requires(const Result value) {
      { value.ok() } noexcept -> std::same_as<bool>;
      { static_cast<bool>(value) } noexcept -> std::same_as<bool>;
      { value.error() } noexcept -> std::same_as<std::string_view>;
      { value.exit_code() } noexcept -> std::same_as<int>;
    };

template <ProductResult Result> bool HasCodeOnlyResultSemantics() noexcept {
  const Result initial{};
  if (initial.ok() || static_cast<bool>(initial) || initial.exit_code() != 1) {
    return false;
  }

  const Result success{rund::ReasonCode::Ok};
  if (!success.ok() || !static_cast<bool>(success) ||
      success.exit_code() != 0) {
    return false;
  }

  const Result timeout{rund::ReasonCode::IoTimedOut};
  return !timeout.ok() && !static_cast<bool>(timeout) &&
         timeout.exit_code() == 1;
}

template <ProductResult Result> bool HasTimedResultSemantics() noexcept {
  const Result initial{};
  if (initial.ok() || static_cast<bool>(initial) || initial.exit_code() != 1) {
    return false;
  }

  const Result success{rund::ReasonCode::Ok};
  if (!success.ok() || !static_cast<bool>(success) ||
      success.exit_code() != 0) {
    return false;
  }

  const Result timeout{rund::ReasonCode::IoTimedOut};
  return timeout.ok() && static_cast<bool>(timeout) && timeout.exit_code() == 0;
}

template <typename... Results> struct ResultTypes {};

template <typename... Results>
consteval bool AllResultsExposeProductContract(ResultTypes<Results...>) {
  return (ProductResult<Results> && ...);
}

template <typename... Results>
bool AllHaveCodeOnlySemantics(ResultTypes<Results...>) noexcept {
  return (HasCodeOnlyResultSemantics<Results>() && ...);
}

template <typename... Results>
bool AllHaveTimedSemantics(ResultTypes<Results...>) noexcept {
  return (HasTimedResultSemantics<Results>() && ...);
}

using OrdinaryResults = ResultTypes<
    rund::net::NonblockingResult, rund::net::ReceiveResult,
    rund::net::SendResult, rund::net::drain::ReadResult,
    rund::net::drain::WriteResult, rund::net::OpenResult, rund::net::BindResult,
    rund::net::ListenResult, rund::net::LocalResult, rund::net::ShutdownResult,
    rund::net::accept::Result, rund::net::accept::Prepared,
    rund::net::CloseResult, rund::net::accept::Drain,
    rund::net::connect::Result, rund::net::datagram::ReceiveResult,
    rund::net::datagram::SendResult, rund::net::option::Result, rund::net::Limits,
    rund::net::ready::Status, rund::net::frame::Result,
    rund::net::frame::ReadResult, rund::net::frame::WriteResult,
    rund::net::server::Result, rund::net::flow::Result>;
using TimedResults =
    ResultTypes<rund::net::ready::Ticket, rund::net::ready::many::Result>;
using TimedCodeResults = ResultTypes<rund::net::ready::many::Result>;

static_assert(AllResultsExposeProductContract(OrdinaryResults{}));
static_assert(AllResultsExposeProductContract(TimedResults{}));
static_assert(!std::is_copy_constructible_v<rund::net::Socket>);
static_assert(!std::is_copy_assignable_v<rund::net::Socket>);
static_assert(std::is_nothrow_move_constructible_v<rund::net::Socket>);
static_assert(std::is_nothrow_move_assignable_v<rund::net::Socket>);
static_assert(sizeof(rund::net::Socket) == 16u);
static_assert(sizeof(rund::net::SocketView) == 16u);
static_assert(std::is_trivially_copyable_v<rund::net::SocketView>);
template <typename Value>
concept CanView =
    requires(Value &&value) { std::forward<Value>(value).view(); };
static_assert(CanView<rund::net::Socket &>);
static_assert(CanView<const rund::net::Socket &>);
static_assert(!CanView<rund::net::Socket &&>);
static_assert(!std::is_copy_constructible_v<rund::net::ready::Ticket>);
static_assert(!std::is_copy_assignable_v<rund::net::ready::Ticket>);
static_assert(std::is_nothrow_move_constructible_v<rund::net::ready::Ticket>);
static_assert(std::is_nothrow_move_assignable_v<rund::net::ready::Ticket>);
static_assert(std::is_trivially_copyable_v<rund::net::ready::many::Result>);
static_assert(!std::is_copy_constructible_v<rund::net::ready::timed::Wait>);
static_assert(!std::is_copy_assignable_v<rund::net::ready::timed::Wait>);
static_assert(
    std::is_nothrow_move_constructible_v<rund::net::ready::timed::Wait>);
static_assert(std::is_nothrow_move_assignable_v<rund::net::ready::timed::Wait>);
static_assert(!std::is_copy_constructible_v<rund::net::ready::many::Wait>);
static_assert(!std::is_copy_assignable_v<rund::net::ready::many::Wait>);
static_assert(
    std::is_nothrow_move_constructible_v<rund::net::ready::many::Wait>);
static_assert(std::is_nothrow_move_assignable_v<rund::net::ready::many::Wait>);
static_assert(std::same_as<
              decltype(std::declval<rund::net::ready::timed::Wait &&>().wait()),
              rund::net::ready::Ticket>);
static_assert(
    std::same_as<
        decltype(std::declval<const rund::net::ready::many::Wait &>().wait()),
        rund::net::ready::many::Result>);
static_assert(!std::is_copy_constructible_v<rund::net::server::Task>);
static_assert(std::is_nothrow_move_constructible_v<rund::net::server::Task>);
static_assert(
    std::same_as<decltype(std::declval<rund::net::server::Task::Awaiter &>()
                              .await_resume()),
                 rund::net::server::Result>);

constexpr rund::net::ready::many::Result kNoEvents{
    rund::ReasonCode::IoTimedOut};
static_assert(kNoEvents.ok() && static_cast<bool>(kNoEvents));
static_assert(kNoEvents.events == 0u && kNoEvents.timed_out());

constexpr rund::net::drain::WriteResult kPartialWrite = [] {
  rund::net::drain::WriteResult result{rund::ReasonCode::Ok};
  result.budget_exhausted = true;
  return result;
}();
static_assert(kPartialWrite.ok() && static_cast<bool>(kPartialWrite));
static_assert(!kPartialWrite.all_written && kPartialWrite.budget_exhausted);

constexpr rund::net::frame::ReadResult kPartialRead = [] {
  rund::net::frame::ReadResult result{rund::ReasonCode::Ok};
  result.header_read = true;
  result.budget_exhausted = true;
  return result;
}();
static_assert(kPartialRead.ok() && static_cast<bool>(kPartialRead));
static_assert(!kPartialRead.complete());

constexpr rund::net::frame::WriteResult kPartialFrameWrite = [] {
  rund::net::frame::WriteResult result{rund::ReasonCode::Ok};
  result.header_written = true;
  result.budget_exhausted = true;
  return result;
}();
static_assert(kPartialFrameWrite.ok() && static_cast<bool>(kPartialFrameWrite));
static_assert(!kPartialFrameWrite.complete());

} // namespace

int RunRuntimeTaskNetContract() {
  TEST_ASSERT(AllHaveCodeOnlySemantics(OrdinaryResults{}));
  TEST_ASSERT(AllHaveTimedSemantics(TimedCodeResults{}));
  const rund::net::ready::Ticket sync_wait =
      std::move(rund::net::ready::timed::read(rund::net::SocketView{},
                                              std::chrono::nanoseconds{-1}))
          .wait();
  TEST_ASSERT(sync_wait.code() == rund::ReasonCode::TimerDurationInvalid);

  TEST_ASSERT(RunNetBasicSyncCase() == 0);
  TEST_ASSERT(RunNetBasicEventCase() == 0);
  TEST_ASSERT(RunNetBasicClosedPeerCase() == 0);
  TEST_ASSERT(RunNetBasicTaskGuardCase() == 0);
  return 0;
}
