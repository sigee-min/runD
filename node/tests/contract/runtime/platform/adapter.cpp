#include "test/assert.hpp"

#include "../../../../src/host/net/native/result.hpp"
#include "../../../../src/runtime/platform/io.hpp"
#include "../../../../src/runtime/platform/net.hpp"
#include "../../../../src/runtime/platform/net/vectored.hpp"
#include "../../../../src/runtime/reactor/platform.hpp"
#include "../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../src/runtime/task/scheduler/reactor/backend.hpp"
#include "../../../../src/runtime/task/scheduler/reactor/model.hpp"

#include <rund/host/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

void VerifyNativeIoResultContract() {
  using rund::node::NativeIoDisposition;
  using rund::node::NativeIoResult;

  static_assert(!std::is_aggregate_v<NativeIoResult>);
  static_assert(!std::is_default_constructible_v<NativeIoResult>);
  static_assert(std::is_trivially_copyable_v<NativeIoResult>);

  constexpr NativeIoResult complete = NativeIoResult::complete(7);
  static_assert(complete.disposition() == NativeIoDisposition::Complete);
  static_assert(complete.value() == 7);
  static_assert(complete.native_error() == 0);

  constexpr NativeIoResult failed = NativeIoResult::failed(EIO);
  static_assert(failed.disposition() == NativeIoDisposition::Failed);
  static_assert(failed.value() == -1);
  static_assert(failed.native_error() == EIO);

  constexpr NativeIoResult invalid = NativeIoResult::invalid_buffer(EINVAL);
  static_assert(invalid.disposition() == NativeIoDisposition::InvalidBuffer);
  static_assert(invalid.value() == -1);
  static_assert(invalid.native_error() == EINVAL);

  constexpr NativeIoResult unsupported = NativeIoResult::unsupported();
  static_assert(unsupported.disposition() == NativeIoDisposition::Unsupported);
  static_assert(unsupported.value() == -1);
  static_assert(unsupported.native_error() == 0);
}

void VerifyNativeFdIdentityContract() {
  using rund::node::NativeFdIdentity;
  using rund::node::NativeFdIdentityDisposition;

  static_assert(!std::is_aggregate_v<NativeFdIdentity>);
  static_assert(!std::is_default_constructible_v<NativeFdIdentity>);
  static_assert(std::is_trivially_copyable_v<NativeFdIdentity>);
  static_assert(sizeof(void *) != 8u || sizeof(NativeFdIdentity) == 32u);

  constexpr NativeFdIdentity invalid = NativeFdIdentity::invalid();
  static_assert(invalid.disposition() == NativeFdIdentityDisposition::Invalid);
  static_assert(invalid.device() == 0u);
  static_assert(invalid.inode() == 0u);
  static_assert(invalid.mode() == 0u);
  static_assert(invalid.type() == 0u);
  static_assert(!invalid.same_socket_object(invalid));

  constexpr NativeFdIdentity described =
      NativeFdIdentity::described(11u, 13u, 15u, 17u);
  static_assert(described.disposition() ==
                NativeFdIdentityDisposition::Described);
  static_assert(described.device() == 11u);
  static_assert(described.inode() == 13u);
  static_assert(described.mode() == 15u);
  static_assert(described.type() == 17u);
  static_assert(described.same_socket_object(
      NativeFdIdentity::described(11u, 13u, 19u, 17u)));
  static_assert(!described.same_socket_object(
      NativeFdIdentity::described(12u, 13u, 15u, 17u)));
  static_assert(!described.same_socket_object(
      NativeFdIdentity::described(11u, 14u, 15u, 17u)));
  static_assert(!described.same_socket_object(
      NativeFdIdentity::described(11u, 13u, 15u, 18u)));
}

#if !defined(RUND_NODE_PLATFORM_UNAVAILABLE)
void VerifyNativeIoProducers() {
  using rund::node::NativeIoDisposition;

  const rund::node::NativeIoResult failed = rund::node::NativeClose(-1);
  TEST_ASSERT(failed.disposition() == NativeIoDisposition::Failed);
  TEST_ASSERT(failed.value() == -1);
  TEST_ASSERT(failed.native_error() == EBADF);

  const rund::node::NativeIoResult invalid = rund::node::NativeRead(
      -1, std::span<std::byte>{static_cast<std::byte *>(nullptr), 1u});
  TEST_ASSERT(invalid.disposition() == NativeIoDisposition::InvalidBuffer);
  TEST_ASSERT(invalid.value() == -1);
  TEST_ASSERT(invalid.native_error() == EINVAL);

  const rund::node::NativeFdIdentity fd_identity =
      rund::node::NativeDescribeFdIdentity(-1);
  TEST_ASSERT(fd_identity.disposition() ==
              rund::node::NativeFdIdentityDisposition::Invalid);
}
#endif

#if defined(RUND_NODE_PLATFORM_UNAVAILABLE)
void AssertUnsupported(const rund::node::NativeIoResult result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::NativeIoDisposition::Unsupported);
  TEST_ASSERT(result.value() == -1);
  TEST_ASSERT(result.native_error() == 0);
}

void AssertUnsupported(const rund::node::NativeCallResult result) {
  TEST_ASSERT(result.value == -1);
  TEST_ASSERT(result.error == 0);
  TEST_ASSERT(result.state == rund::node::NativeCallState::Unsupported);
}

void AssertUnsupported(const rund::node::NativeAddressResult &result) {
  AssertUnsupported(result.call);
  TEST_ASSERT(!result.address);
  TEST_ASSERT(result.address.family() == rund::net::Family::None);
  TEST_ASSERT(result.address.port() == 0u);
  TEST_ASSERT(result.address.bytes().empty());
}

void AssertPrepared(const rund::node::ReactorPlatformOpResult result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::ReactorPlatformOpDisposition::Success);
  TEST_ASSERT(result.platform_error() == 0);
}

void AssertUnavailable(const rund::node::ReactorPlatformOpResult result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::ReactorPlatformOpDisposition::BackendUnavailable);
  TEST_ASSERT(result.platform_error() == 0);
}

void AssertUnavailable(const rund::node::ReactorPlatformBatchResult result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::ReactorPlatformBatchDisposition::BackendUnavailable);
  TEST_ASSERT(result.platform_error() == 0);
  TEST_ASSERT(result.failed_index() == 0u);
}

void AssertUnavailable(const rund::node::ReactorPlatformPollResult &result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::ReactorPlatformPollDisposition::BackendUnavailable);
  TEST_ASSERT(result.platform_error() == 0);
}

void AssertUnavailable(const rund::node::BatchIoProbeResult result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::BatchIoProbeDisposition::BackendUnavailable);
  TEST_ASSERT(result.platform_error() == 0);
}

void AssertUnavailable(const rund::node::ReactorApplyResult result) {
  TEST_ASSERT(result.disposition() ==
              rund::node::ReactorApplyDisposition::BackendUnavailable);
  TEST_ASSERT(!result.invalid_change().valid());
}

void AssertNativeProjection(const rund::node::NativeCallState state,
                            const rund::ReasonCode code,
                            const rund::host::Status status) {
  const rund::node::NativeCallResult result{.state = state};
  TEST_ASSERT(rund::net::CodeForNative(result) == code);
  TEST_ASSERT(rund::net::StatusForNative(result) == status);
}

void VerifyNativeProjection() {
  using rund::ReasonCode;
  using rund::host::Status;
  using rund::node::NativeCallState;

  AssertNativeProjection(NativeCallState::Complete, ReasonCode::Ok, Status::Ok);
  AssertNativeProjection(NativeCallState::InProgress, ReasonCode::Ok,
                         Status::Ok);
  AssertNativeProjection(NativeCallState::Failed, ReasonCode::IoSyscallFailed,
                         Status::SyscallFailed);
  AssertNativeProjection(NativeCallState::InvalidInput, ReasonCode::TaskInvalid,
                         Status::Invalid);
  AssertNativeProjection(NativeCallState::Unsupported,
                         ReasonCode::IoUnsupported, Status::Unsupported);
  AssertNativeProjection(NativeCallState::WouldBlock, ReasonCode::IoWouldBlock,
                         Status::WouldBlock);
}

void VerifyUnavailableNativeSurface() {
  using namespace rund::node;

  VerifyNativeProjection();

  std::array<std::byte, 1u> bytes{};
  const rund::net::Address address{};
  const std::array<rund::net::batch::Buffer, 1u> mutable_slices{
      rund::net::batch::Buffer{.data = bytes.data(), .size = bytes.size()}};
  const std::array<rund::net::batch::Slice, 1u> slices{
      rund::net::batch::Slice{.data = bytes.data(), .size = bytes.size()}};
  const nativeio::VectoredBatch recv_batch = nativeio::PrepareSlices(
      std::span<const rund::net::batch::Buffer>{mutable_slices},
      mutable_slices.size(), bytes.size());
  const nativeio::VectoredBatch send_batch = nativeio::PrepareSlices(
      std::span<const rund::net::batch::Slice>{slices}, slices.size(),
      bytes.size());
  TEST_ASSERT(recv_batch.valid);
  TEST_ASSERT(send_batch.valid);
  static_assert(std::is_same_v<decltype(NativeRecvVectored(0, recv_batch)),
                               NativeCallResult>);
  static_assert(std::is_same_v<decltype(NativeSendVectored(0, send_batch)),
                               NativeCallResult>);
  TEST_ASSERT(recv_batch.admitted_bytes == 1u);
  TEST_ASSERT(send_batch.admitted_bytes == 1u);

  TEST_ASSERT(!NativeFdValid(0));
  const NativeFdIdentity fd_identity = NativeDescribeFdIdentity(0);
  TEST_ASSERT(fd_identity.disposition() ==
              NativeFdIdentityDisposition::Invalid);
  TEST_ASSERT(fd_identity.device() == 0u);
  TEST_ASSERT(fd_identity.inode() == 0u);
  TEST_ASSERT(fd_identity.mode() == 0u);
  TEST_ASSERT(fd_identity.type() == 0u);
  TEST_ASSERT(!NativeIsNonblockingFd(0));
  AssertUnsupported(NativeSetNonblockingFd(0, true));
  AssertUnsupported(NativeRead(0, bytes));
  AssertUnsupported(NativeWrite(0, bytes));
  AssertUnsupported(NativePread(0, bytes, 0u));
  AssertUnsupported(NativeOpen("unavailable", 0, 0u));
  AssertUnsupported(NativeClose(0));

  AssertUnsupported(
      NativeSocket(rund::net::Family::IPv4, rund::net::Transport::Stream));
  AssertUnsupported(NativeBind(0, address));
  AssertUnsupported(NativeListen(0, 1));
  AssertUnsupported(NativeGetSockName(0));
  AssertUnsupported(
      NativeShutdown(0, rund::net::ShutdownMode::ReadWrite));
  AssertUnsupported(NativePrepareSocketSend(0));
  AssertUnsupported(NativeRecv(0, bytes));
  AssertUnsupported(NativeSend(0, bytes));
  AssertUnsupported(NativeTryRecv(0, bytes));
  AssertUnsupported(NativeTrySend(0, bytes));
  AssertUnsupported(NativeRecvFrom(0, bytes));
  AssertUnsupported(NativeSendTo(0, bytes, address));
  AssertUnsupported(NativeRecvVectored(0, recv_batch));
  AssertUnsupported(NativeSendVectored(0, send_batch));
  TEST_ASSERT(recv_batch.admitted_bytes == 1u);
  TEST_ASSERT(send_batch.admitted_bytes == 1u);
  AssertUnsupported(NativeAccept(0));
  AssertUnsupported(NativeConnect(0, address));
  AssertUnsupported(NativeGetSocketError(0, address));
  AssertUnsupported(NativeSetSocketOption(
      0, rund::net::option::Name::ReuseAddress,
      rund::net::option::Value{.flag = true}));
  AssertUnsupported(
      NativeGetSocketOption(0, rund::net::option::Name::ReuseAddress));

  ReactorPlatform platform{};
  AssertPrepared(PrepareReactorPlatform(platform, 8u));
  TEST_ASSERT(platform.state == nullptr);
  AssertUnavailable(OpenReactorPlatform(platform));
  AssertUnavailable(AddReactorPlatformInterest(
      platform, ReactorHandleFromPublic(0), ReactorInterest::Read));
  AssertUnavailable(ModifyReactorPlatformInterest(
      platform, ReactorHandleFromPublic(0), ReactorInterest::Write));
  AssertUnavailable(
      RemoveReactorPlatformInterest(platform, ReactorHandleFromPublic(0)));
  AssertUnavailable(ApplyReactorPlatformChanges(platform, nullptr, 0u));
  std::vector<ReactorPlatformReady> poll_ready{ReactorPlatformReady{}};
  AssertUnavailable(PollReactorPlatform(platform, 0, 1u, poll_ready));
  TEST_ASSERT(poll_ready.empty());
  std::vector<BatchIoReady> ready{BatchIoReady{}};
  const BatchIoPollRequest request{
      .index = 0u,
      .handle = ReactorHandleFromPublic(0),
      .interest = ReactorInterest::Read,
  };
  AssertUnavailable(ProbeReactorPlatformNow(platform, &request, 1u, ready));
  TEST_ASSERT(ready.empty());

  ReactorRuntime scheduler_reactor{};
  ::rund::detail::task::StatStorage scheduler_stats{};
  AssertUnavailable(
      ReactorBackendApplyChanges(scheduler_reactor, scheduler_stats));
  ReactorCloseRuntime(scheduler_reactor);

  const ReactorPlatformHandleIdentity handle_identity =
      DescribeReactorPlatformHandle(ReactorHandleFromPublic(0));
  TEST_ASSERT(handle_identity.disposition() ==
              ReactorPlatformHandleIdentityDisposition::Invalid);
  TEST_ASSERT(handle_identity.device() == 0u);
  TEST_ASSERT(handle_identity.inode() == 0u);
  TEST_ASSERT(handle_identity.mode() == 0u);
  TEST_ASSERT(RetainReactorPlatformHandle(ReactorHandleFromPublic(0)) ==
              kInvalidReactorHandle);
  ReleaseReactorPlatformHandle(kInvalidReactorHandle);
  CloseReactorPlatform(platform);
  TEST_ASSERT(platform.state == nullptr);
}
#endif

} // namespace

int RunRuntimePlatformAdapterContract() {
  using rund::ReasonCode;

  VerifyNativeIoResultContract();
  VerifyNativeFdIdentityContract();
#if !defined(RUND_NODE_PLATFORM_UNAVAILABLE)
  VerifyNativeIoProducers();
#endif

  static_assert(std::is_same_v<decltype(rund::host::io::OpenOptions{}.mode),
                               std::uint32_t>);
  TEST_ASSERT(std::string_view{
                  rund::ReasonString(ReasonCode::ReactorBackendUnavailable)} ==
              "reactor_backend_unavailable");
  TEST_ASSERT(std::string_view{rund::ReasonString(ReasonCode::IoUnsupported)} ==
              "io_unsupported");

#if defined(RUND_NODE_PLATFORM_UNAVAILABLE)
  VerifyUnavailableNativeSurface();

  std::array<std::byte, 1u> bytes{};
  rund::host::io::Fd admitted = rund::host::io::take_native_fd(0);
  TEST_ASSERT(
      rund::host::io::read_some_blocking(admitted.view(), bytes).code() ==
      ReasonCode::IoUnsupported);
  TEST_ASSERT(
      rund::host::io::write_some_blocking(admitted.view(), bytes).code() ==
      ReasonCode::IoUnsupported);
  TEST_ASSERT(rund::host::io::pread_some(admitted.view(), bytes, 0u).code() ==
              ReasonCode::IoUnsupported);
  rund::host::io::Fd close_probe = rund::host::io::take_native_fd(0);
  TEST_ASSERT(close_probe.close().code() == ReasonCode::IoUnsupported);

  const rund::host::io::OpenResult opened =
      rund::host::io::open_file("unavailable", rund::host::io::OpenOptions{});
  TEST_ASSERT(!opened);
  TEST_ASSERT(opened.code() == ReasonCode::IoUnsupported);
  TEST_ASSERT(opened.error() == "io_unsupported");

  rund::host::io::WriteResult recorded_write{};
  rund::task::Status recorded_join{};
  const rund::SessionConfig record_config{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
              .host_io_capacity = 1u,
              .host_event_capacity = 1u,
              .host_payload_capacity_bytes = bytes.size(),
          },
  };
  const rund::Session::Result recorded = rund::run(record_config, [&] {
    auto body = [&]() -> rund::task::Task<void> {
      recorded_write =
          co_await rund::host::io::write_some(admitted.view(), bytes);
    };
    const rund::task::Handle task =
        rund::task::spawn("unavailable-hostio-record", body());
    recorded_join = rund::task::join(task);
  });
  TEST_ASSERT(recorded);
  TEST_ASSERT(recorded_join);
  TEST_ASSERT(!recorded_write);
  TEST_ASSERT(recorded_write.code() == ReasonCode::IoUnsupported);
  TEST_ASSERT(recorded.events().size() == 1u);
  TEST_ASSERT(recorded.events()[0].status == rund::host::Status::Unsupported);
  TEST_ASSERT(recorded.tasks().external_parks() == 1u);
  TEST_ASSERT(recorded.tasks().external_wakes() == 1u);

  rund::net::OpenResult socket{};
  const rund::Session::Result socket_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {.host_event_capacity = 1u},
      },
      [&] { socket = rund::net::open(rund::net::OpenOptions{}); });
  TEST_ASSERT(socket_report);
  TEST_ASSERT(!socket);
  TEST_ASSERT(socket.code() == ReasonCode::IoUnsupported);
  TEST_ASSERT(socket.error() == "io_unsupported");
  TEST_ASSERT(socket.native_error == 0);
  TEST_ASSERT(socket_report.events().size() == 1u);
  TEST_ASSERT(socket_report.events()[0].kind ==
              rund::host::EventKind::NetSocket);
  TEST_ASSERT(socket_report.events()[0].status ==
              rund::host::Status::Unsupported);
  TEST_ASSERT(socket_report.events()[0].native_errno == 0);

  rund::task::IoResult readiness{};
  rund::task::Status joined{};
  rund::host::io::Fd ready_fd = rund::host::io::take_native_fd(0);
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .reactor_wait_capacity = 1u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          readiness = co_await rund::host::io::readable(ready_fd.view());
          co_return;
        };
        const rund::task::Handle task =
            rund::task::spawn("unavailable-reactor", body());
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined);
  TEST_ASSERT(!readiness);
  TEST_ASSERT(readiness.code() == ReasonCode::ReactorBackendUnavailable);
  TEST_ASSERT(readiness.error() == "reactor_backend_unavailable");
#endif

  return 0;
}
