#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/host.hpp>
#include <rund/net.hpp>
#include <rund/replay.hpp>
#include <rund/session.hpp>
#include <rund/storage.hpp>
#include <rund/task.hpp>

#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

namespace {

struct Source final {
  std::uint64_t operator()(rund::replay::Writer &) { return 1u; }
};

struct VoidSource final {
  void operator()(rund::replay::Writer &) {}
};

struct Restore final {
  rund::replay::Restore operator()(std::span<const std::byte>) {
    return rund::replay::Restore::Restored;
  }
};

struct Callback final {
  void operator()(rund::replay::Context &) {}
};

struct ReadDrain final {
  bool operator()(std::span<const std::byte>) noexcept { return true; }
};

struct WriteDrain final {
  bool operator()(std::uint64_t, rund::net::SendResult) noexcept {
    return true;
  }
};

template <class Binding, class InputSource>
concept BindsInput = requires(Binding &binding, InputSource &source) {
  {
    binding.input(rund::replay::Input{.id = 1u, .schema = 1u}, source)
  } -> std::same_as<rund::replay::Channel>;
};

template <class Binding, class InputSource>
concept BindsTemporaryInput = requires(Binding &binding, InputSource &&source) {
  binding.input(rund::replay::Input{.id = 1u, .schema = 1u}, std::move(source));
};

template <class Channel>
concept ReadsCanonicalInput =
    requires(const Channel &channel, rund::replay::Context &context) {
      { channel.read(context) } -> std::same_as<rund::replay::Value>;
    };

template <class Channel>
concept CreatesChoice =
    requires(const Channel &channel, std::span<const std::byte> bytes) {
      {
        channel.choice(std::uint64_t{1u}, bytes)
      } -> std::same_as<rund::replay::Choice>;
    };

template <class Channel>
concept AcceptsCallerSequence =
    requires(const Channel &channel, rund::replay::Context &context) {
      channel.read(context, std::uint64_t{1u});
    };

template <class Binding>
concept OwnsStateBoundary =
    requires(const Binding &binding, const rund::replay::Record &record,
             const rund::replay::Checkpoint &checkpoint,
             rund::replay::History &history, rund::replay::Record &&completed,
             std::span<const std::byte> bytes) {
      {
        binding.checkpoint(record, bytes)
      } -> std::same_as<rund::replay::Checkpoint>;
      {
        binding.advance(checkpoint, record, bytes)
      } -> std::same_as<rund::replay::Checkpoint>;
      { binding.resume(checkpoint) } -> std::same_as<rund::replay::Resume>;
      {
        binding.append(history, std::move(completed), bytes)
      } -> std::same_as<rund::replay::Append>;
    };

template <class Checkpoint>
concept ExecutesDirectly =
    requires(const Checkpoint &checkpoint, rund::Session &session,
             Restore &restore, Callback callback) {
      checkpoint.record(session, restore, std::move(callback));
    };

template <class Values>
concept SupportsPushBack =
    requires(Values &values) { values.push_back(rund::telemetry::Finding{}); };

template <class Values>
concept SupportsAppend =
    requires(Values &values) { values.append(rund::telemetry::Finding{}); };

template <class Values>
concept SupportsMemberAppend = requires(Values &values) {
  values.append(rund::telemetry::Action::ReuseJob);
};

template <class Session, class Job>
concept RunsCompute =
    requires(Session &session, Job &job) { session.compute(job); };

template <class Selector>
concept OpensCompute =
    requires(Selector selector) { rund::compute::open(selector); };

template <class Selector>
concept BuildsCompute =
    requires(Selector selector) { rund::compute::on(selector); };

template <class Input>
concept DrainsReadable =
    requires(Input &&input, std::span<std::byte> bytes, ReadDrain &drain) {
      {
        rund::net::drain::read(std::forward<Input>(input), bytes,
                               rund::net::drain::Budget{}, drain)
      } -> std::same_as<rund::net::drain::ReadResult>;
    };

template <class Input>
concept DrainsWritable = requires(
    Input &&input, std::span<const std::byte> bytes, WriteDrain &drain) {
  {
    rund::net::drain::write(std::forward<Input>(input), bytes,
                            rund::net::drain::Budget{}, drain)
  } -> std::same_as<rund::net::drain::WriteResult>;
};

template <class Owner>
concept BorrowsSocketView = requires(Owner &owner) {
  { owner.view() } -> std::same_as<rund::net::SocketView>;
};

template <class Owner>
concept BorrowsSocketViewFromTemporary =
    requires(Owner &&owner) { std::move(owner).view(); };

template <class View>
concept ClosesSocket = requires(View &view) { view.close(); };

template <class Task>
concept ExposesTaskHandle = requires(const Task &task) { task.handle(); };

template <class Task>
concept ReleasesTaskFrame = requires(Task &task) { task.release(); };

template <class Task>
concept DestroysTaskFrame = requires(Task &task) { task.destroy(); };

template <class Task>
concept ExposesTaskHandleType = requires { typename Task::handle_type; };

using ComputeJob = rund::compute::Job<std::int32_t(std::int32_t)>;
using ComputePipeline = rund::compute::Pipeline;
using ValueResult = rund::compute::Result<std::int32_t>;
using TaskResult = rund::task::Result<std::int32_t>;
using PublicTask = rund::task::Task<void>;
using PublicTaskFrame = std::coroutine_handle<PublicTask::promise_type>;
using StreamReceive = decltype(rund::net::receive(rund::net::SocketView{},
                                                  std::span<std::byte>{}));
using StreamReceiveAwaiter =
    decltype(std::declval<StreamReceive &&>().operator co_await());
using StreamSend = decltype(rund::net::send(rund::net::SocketView{},
                                            std::span<const std::byte>{}));
using StreamSendAwaiter =
    decltype(std::declval<StreamSend &&>().operator co_await());
using DatagramReceive = decltype(rund::net::datagram::receive(
    rund::net::SocketView{}, std::span<std::byte>{}));
using DatagramReceiveAwaiter =
    decltype(std::declval<DatagramReceive &&>().operator co_await());
using DatagramSend = decltype(rund::net::datagram::send(
    rund::net::SocketView{}, std::span<const std::byte>{},
    rund::net::Address{}));
using DatagramSendAwaiter =
    decltype(std::declval<DatagramSend &&>().operator co_await());

static_assert(!std::same_as<ValueResult, TaskResult>);
static_assert(OpensCompute<rund::compute::Target>);
static_assert(BuildsCompute<rund::compute::Target>);
static_assert(!OpensCompute<rund::compute::Device>);
static_assert(BuildsCompute<rund::compute::Device>);
static_assert(!OpensCompute<rund::compute::ProgramCache>);
static_assert(!BuildsCompute<rund::compute::ProgramCache>);
static_assert(!OpensCompute<rund::compute::Backend>);
static_assert(!BuildsCompute<rund::compute::Backend>);
static_assert(
    !std::constructible_from<rund::compute::Target, rund::compute::Backend>);
static_assert(rund::compute::Target::cpu().backend() ==
              rund::compute::Backend::Cpu);
static_assert(rund::compute::Target::metal().backend() ==
              rund::compute::Backend::Metal);
static_assert(rund::compute::Target::vulkan().backend() ==
              rund::compute::Backend::Vulkan);
static_assert(rund::compute::Target::cpu(3u).workers() == 3u);
static_assert(std::same_as<StreamReceive, rund::net::Receive>);
static_assert(std::same_as<StreamSend, rund::net::Send>);
static_assert(std::same_as<DatagramReceive, rund::net::datagram::Receive>);
static_assert(std::same_as<DatagramSend, rund::net::datagram::Send>);
static_assert(!std::copy_constructible<StreamReceive>);
static_assert(!std::is_copy_assignable_v<StreamReceive>);
static_assert(std::is_nothrow_move_constructible_v<StreamReceive>);
static_assert(std::is_nothrow_move_assignable_v<StreamReceive>);
static_assert(!std::copy_constructible<StreamSend>);
static_assert(!std::is_copy_assignable_v<StreamSend>);
static_assert(std::is_nothrow_move_constructible_v<StreamSend>);
static_assert(std::is_nothrow_move_assignable_v<StreamSend>);
static_assert(!std::copy_constructible<DatagramReceive>);
static_assert(!std::is_copy_assignable_v<DatagramReceive>);
static_assert(std::is_nothrow_move_constructible_v<DatagramReceive>);
static_assert(std::is_nothrow_move_assignable_v<DatagramReceive>);
static_assert(!std::copy_constructible<DatagramSend>);
static_assert(!std::is_copy_assignable_v<DatagramSend>);
static_assert(std::is_nothrow_move_constructible_v<DatagramSend>);
static_assert(std::is_nothrow_move_assignable_v<DatagramSend>);
static_assert(std::same_as<
              decltype(std::declval<StreamReceiveAwaiter &>().await_resume()),
              rund::net::ReceiveResult>);
static_assert(
    std::same_as<decltype(std::declval<StreamSendAwaiter &>().await_resume()),
                 rund::net::SendResult>);
static_assert(std::same_as<
              decltype(std::declval<DatagramReceiveAwaiter &>().await_resume()),
              rund::net::datagram::ReceiveResult>);
static_assert(
    std::same_as<decltype(std::declval<DatagramSendAwaiter &>().await_resume()),
                 rund::net::datagram::SendResult>);
static_assert(std::same_as<decltype(ValueResult::success(1)), ValueResult>);
static_assert(std::same_as<decltype(TaskResult::success(1)), TaskResult>);
static_assert(noexcept(std::declval<const ValueResult &>().code()));
static_assert(noexcept(std::declval<const ValueResult &>().reason()));
static_assert(noexcept(std::declval<const ValueResult &>().error()));
static_assert(noexcept(std::declval<const ValueResult &>().exit_code()));
static_assert(noexcept(std::declval<const TaskResult &>().code()));
static_assert(noexcept(std::declval<const TaskResult &>().error()));
static_assert(noexcept(std::declval<const TaskResult &>().exit_code()));
static_assert(!std::copy_constructible<PublicTask>);
static_assert(std::move_constructible<PublicTask>);
static_assert(!ExposesTaskHandle<PublicTask>);
static_assert(!ReleasesTaskFrame<PublicTask>);
static_assert(!DestroysTaskFrame<PublicTask>);
static_assert(!ExposesTaskHandleType<PublicTask>);
static_assert(!std::constructible_from<PublicTask, PublicTaskFrame>);

static_assert(std::same_as<decltype(std::declval<rund::Session &>().compute(
                               std::declval<ComputeJob &>())),
                           rund::compute::Request>);
static_assert(std::same_as<decltype(std::declval<rund::Session &>().compute(
                               std::declval<ComputePipeline &>())),
                           rund::compute::Request>);
static_assert(
    std::same_as<decltype(std::declval<rund::compute::Request &>().submit()),
                 rund::compute::Submission>);
static_assert(
    std::same_as<
        decltype(std::declval<const rund::compute::Submission &>().poll()),
        rund::compute::Poll>);
static_assert(
    std::same_as<
        decltype(std::declval<const rund::compute::Submission &>().wait()),
        rund::compute::Completion>);
static_assert(!std::is_copy_constructible_v<rund::compute::Request>);
static_assert(std::is_nothrow_move_constructible_v<rund::compute::Request>);
static_assert(!std::is_copy_constructible_v<rund::compute::Submission>);
static_assert(std::is_nothrow_move_constructible_v<rund::compute::Submission>);
static_assert(std::is_same_v<decltype(rund::Session::Status{}.state()),
                             rund::SessionState>);
static_assert(std::is_same_v<decltype(rund::Session::Result{}.memory()),
                             const rund::PreparedMemory &>);
static_assert(std::is_same_v<decltype(rund::Session::Result{}.events()),
                             std::span<const rund::host::Event>>);
static_assert(std::is_same_v<decltype(rund::Session::Result{}.trace()),
                             const rund::Trace &>);
static_assert(
    std::is_same_v<decltype(std::declval<const rund::Session &>().resources()),
                   rund::Resources>);
static_assert(std::is_trivially_copyable_v<rund::StableHash>);
static_assert(std::is_same_v<decltype(rund::net::BindResult{}.address_hash),
                             rund::StableHash>);
static_assert(
    std::same_as<decltype(rund::net::flow::reserve(
                     rund::net::flow::State{}, rund::net::flow::Limit{}, 1u)),
                 rund::net::flow::Result>);
static_assert(DrainsReadable<rund::net::ready::Ticket>);
static_assert(DrainsWritable<rund::net::ready::Ticket>);
static_assert(!DrainsReadable<rund::net::SocketView>);
static_assert(!DrainsWritable<rund::net::SocketView>);
static_assert(BorrowsSocketView<rund::net::Socket>);
static_assert(!BorrowsSocketViewFromTemporary<rund::net::Socket>);
static_assert(!ClosesSocket<rund::net::SocketView>);
static_assert(
    !std::constructible_from<rund::net::SocketView, void *, std::uint64_t>);

static_assert(BindsInput<rund::replay::Binding, Source>);
static_assert(!BindsInput<rund::replay::Binding, VoidSource>);
static_assert(!BindsTemporaryInput<rund::replay::Binding, Source>);
static_assert(
    std::constructible_from<rund::replay::Binding, std::uint64_t, Restore &>);
static_assert(
    !std::constructible_from<rund::replay::Binding, std::uint64_t, Restore>);
static_assert(ReadsCanonicalInput<rund::replay::Channel>);
static_assert(CreatesChoice<rund::replay::Channel>);
static_assert(OwnsStateBoundary<rund::replay::Binding>);
static_assert(
    !AcceptsCallerSequence<rund::replay::Channel>,
    "Channel owns canonical transcript sequencing; callers provide no index");
static_assert(!std::is_aggregate_v<rund::replay::Choice>);
static_assert(!ExecutesDirectly<rund::replay::Checkpoint>);
static_assert(
    std::same_as<decltype(rund::replay::Storage{}.max_allocated_bytes),
                 std::uint64_t>);
static_assert(std::same_as<decltype(rund::replay::Storage{}.minimum_free_bytes),
                           std::uint64_t>);
static_assert(std::same_as<decltype(rund::replay::Limits{}.max_entries),
                           std::uint64_t>);
static_assert(rund::replay::Limits{}.max_bytes ==
              2ull * 1024ull * 1024ull * 1024ull);
static_assert(rund::replay::Limits{}.max_entries == 4ull * 1024ull * 1024ull);
static_assert(rund::replay::Limits{}.max_payload_bytes ==
              1024ull * 1024ull * 1024ull);
static_assert(rund::replay::Limits{}.max_state_bytes ==
              64ull * 1024ull * 1024ull);

static_assert(std::copy_constructible<rund::storage::Budget>);
static_assert(std::copyable<rund::storage::Budget>);
static_assert(!std::copy_constructible<rund::storage::Reservation>);
static_assert(std::movable<rund::storage::Reservation>);
static_assert(
    std::same_as<decltype(std::declval<const rund::storage::Budget &>().child(
                     std::uint64_t{1u})),
                 rund::storage::Budget>);
static_assert(
    std::same_as<decltype(std::declval<const rund::storage::Budget &>().reserve(
                     std::uint64_t{1u})),
                 rund::storage::Reservation>);
static_assert(std::same_as<
              decltype(std::declval<const rund::storage::Budget &>().report()),
              rund::storage::Report>);
static_assert(std::same_as<decltype(std::declval<rund::storage::Reservation &>()
                                        .commit(rund::storage::Usage{})),
                           rund::storage::Status>);
static_assert(std::same_as<
              decltype(std::declval<rund::storage::Reservation &>().refund()),
              rund::storage::Status>);

static_assert(rund::telemetry::Findings::Capacity == 5u);
static_assert(std::is_trivially_copyable_v<rund::telemetry::Finding>);
static_assert(!SupportsPushBack<rund::telemetry::Findings>);
static_assert(!SupportsAppend<rund::telemetry::Findings>);
static_assert(
    !SupportsMemberAppend<rund::telemetry::Members<rund::telemetry::Action>>);

static_assert(RunsCompute<rund::Session, ComputeJob>);

} // namespace

int main() {
  Source source{};
  Restore restore{};
  rund::replay::Binding input{};
  const auto channel =
      input.input(rund::replay::Input{.id = 1u, .schema = 1u}, source);
  if (!channel) {
    return channel.exit_code();
  }
  rund::replay::Binding state{1u, restore};
  if (!state) {
    return state.exit_code();
  }
  rund::telemetry::Findings findings{};
  const rund::replay::Storage storage{};
  const bool telemetry_text =
      rund::telemetry::name(rund::telemetry::Accuracy::Exact) == "exact" &&
      rund::telemetry::name(rund::telemetry::Accuracy::Unavailable) ==
          "unavailable" &&
      rund::telemetry::name(rund::telemetry::Accuracy::Saturated) ==
          "saturated" &&
      rund::telemetry::name(rund::telemetry::Reference::None) == "none" &&
      rund::telemetry::name(rund::telemetry::Reference::ReuseEvents) ==
          "reuse-events" &&
      rund::telemetry::name(rund::telemetry::Reference::RetainedBytes) ==
          "retained-bytes" &&
      rund::telemetry::name(rund::telemetry::Reference::QueueCapacity) ==
          "queue-capacity" &&
      rund::telemetry::name(rund::telemetry::Reference::PhaseTotal) ==
          "phase-total";
  return state.checkpointable() && findings.empty() && telemetry_text &&
                 storage.max_allocated_bytes ==
                     2ull * 1024ull * 1024ull * 1024ull &&
                 storage.minimum_free_bytes == 0u
             ? 0
             : 2;
}
