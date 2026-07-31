#pragma once

#include <rund/rund.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace package_blackbox {

[[nodiscard]] inline int Mismatch(const char *const reason) {
  std::fprintf(stderr, "package blackbox failed: %s\n", reason);
  return 2;
}

template <class Operation>
[[nodiscard]] int Finish(rund::Session &session, Operation &&operation) {
  const int outcome = std::forward<Operation>(operation)();
  const rund::Session::Status closed = session.close();
  if (outcome != 0 && outcome != 2) {
    return outcome;
  }
  if (!closed) {
    return closed.exit_code();
  }
  return outcome;
}

struct Artifact {
  rund::replay::Save saved;
  std::vector<std::byte> bytes{};
};

template <typename Value> [[nodiscard]] Artifact Persist(const Value &value) {
  std::vector<std::byte> artifact{};
  const rund::replay::Save saved =
      value.save([&artifact](const std::span<const std::byte> bytes) noexcept {
        try {
          artifact.insert(artifact.end(), bytes.begin(), bytes.end());
          return true;
        } catch (...) {
          return false;
        }
      });
  return Artifact{saved, std::move(artifact)};
}

struct TaskRun {
  std::uint32_t ran = 0u;
  rund::task::Status operation{};
  rund::task::Status joined{};
};

inline rund::task::Task<void> RunTask(TaskRun &run) {
  ++run.ran;
  const rund::task::Status yielded = co_await rund::task::yield();
  if (!yielded) {
    run.operation = yielded;
    co_return;
  }
  const rund::task::Status slept =
      co_await rund::task::sleep(std::chrono::nanoseconds{1});
  run.operation = slept;
}

inline void RunBlackboxTask(TaskRun &run) {
  const rund::task::Handle task =
      rund::task::spawn("blackbox-task", RunTask(run));
  if (!task) {
    run.joined = rund::task::Status::fail(task.code());
    return;
  }
  run.joined = rund::task::join(task);
}

static_assert(!std::is_copy_constructible_v<rund::host::io::Fd>);
static_assert(std::is_nothrow_move_constructible_v<rund::host::io::Fd>);
static_assert(std::is_trivially_copyable_v<rund::host::io::FdView>);
static_assert(!std::is_copy_constructible_v<rund::host::io::ReadOp>);
static_assert(std::is_nothrow_move_constructible_v<rund::host::io::ReadOp>);
static_assert(!std::is_copy_constructible_v<rund::host::io::WriteOp>);
static_assert(std::is_nothrow_move_constructible_v<rund::host::io::WriteOp>);
static_assert(
    std::is_same_v<decltype(rund::replay::Mismatch{}.field), std::string_view>);
static_assert(
    std::is_same_v<decltype(rund::replay::Trace{}.code), rund::TraceCode>);
static_assert(std::is_same_v<decltype(rund::replay::Trace{}.snapshot_code),
                             rund::ReasonCode>);
static_assert(
    std::is_same_v<decltype(rund::Session::Snapshot{}.code), rund::ReasonCode>);
static_assert(noexcept(
    std::declval<const rund::replay::Diff &>().mismatch(std::size_t{0u})));

inline rund::task::Task<void> TransferHostIo(
    const rund::host::io::FdView writer, const rund::host::io::FdView reader,
    const std::span<const std::byte> out, const std::span<std::byte> in,
    rund::host::io::WriteResult &written, rund::host::io::ReadResult &read) {
  written = co_await rund::host::io::write_some(writer, out);
  if (!written) {
    co_return;
  }
  read = co_await rund::host::io::read_some(reader, in);
}

[[nodiscard]] int CheckRunReplay();
[[nodiscard]] int CheckMathAndEvidence();
[[nodiscard]] int CheckCluster();
[[nodiscard]] int CheckNetwork();

} // namespace package_blackbox
