#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/host.hpp>
#include <rund/net.hpp>
#include <rund/session.hpp>
#include <rund/task.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

int main() {
  static_assert(std::is_same_v<
                decltype(std::declval<const rund::compute::Poll &>().reason()),
                rund::compute::Reason>);
  static_assert(std::is_same_v<
                decltype(std::declval<const rund::compute::Poll &>().code()),
                rund::compute::Code>);
  static_assert(noexcept(std::declval<const rund::compute::Poll &>().reason()));
  static_assert(
      noexcept(std::declval<const rund::compute::Completion &>().reason()));
  static_assert(
      noexcept(std::declval<const rund::compute::Completion &>().exit_code()));
  static_assert(std::is_same_v<decltype(rund::Session::Result{}.code()),
                               rund::ReasonCode>);
  static_assert(std::is_same_v<decltype(rund::Session::Result{}.tasks()),
                               const rund::task::Stats &>);
  static_assert(std::is_same_v<decltype(rund::Session::Result{}.observations()),
                               std::span<const rund::task::Observation>>);
  static_assert(sizeof(rund::task::Status) == sizeof(rund::ReasonCode));
  static_assert(sizeof(rund::task::Handle) == 32u);
  static_assert(std::is_trivially_copyable_v<rund::task::Status>);
  static_assert(std::is_trivially_copyable_v<rund::task::Handle>);
  static_assert(std::is_same_v<decltype(rund::Session::Result{}.memory()),
                               const rund::PreparedMemory &>);
  static_assert(std::is_same_v<decltype(rund::Session::Result{}.events()),
                               std::span<const rund::host::Event>>);
  static_assert(std::is_same_v<decltype(rund::Session::Result{}.trace()),
                               const rund::Trace &>);
  static_assert(noexcept(std::declval<const rund::Session::Result &>().ok()));
  static_assert(
      noexcept(std::declval<const rund::Session::Result &>().trace_hash()));
  static_assert(!std::is_copy_constructible_v<rund::Session::Result>);
  static_assert(std::is_nothrow_move_constructible_v<rund::Session::Result>);
  static_assert(std::is_same_v<decltype(rund::Session::Status{}.code()),
                               rund::ReasonCode>);
  static_assert(std::is_same_v<decltype(rund::Session::Status{}.state()),
                               rund::SessionState>);
  static_assert(std::is_same_v<
                decltype(std::declval<const rund::Session &>().resources()),
                rund::Resources>);
  static_assert(std::is_trivially_copyable_v<rund::StableHash>);
  static_assert(std::is_same_v<decltype(rund::net::BindResult{}.address_hash),
                               rund::StableHash>);
  static_assert(std::is_same_v<decltype(rund::net::accept::Result{}.peer_hash),
                               rund::StableHash>);
  static_assert(
      std::is_same_v<decltype(rund::net::datagram::ReceiveResult{}.peer_hash),
                     rund::StableHash>);
  static_assert(!std::is_constructible_v<rund::Session::Status,
                                         rund::ReasonCode, rund::SessionState>);
  static_assert(std::is_same_v<decltype(rund::telemetry::Event{}.session),
                               std::uint64_t>);
  static_assert(std::is_trivially_copyable_v<rund::telemetry::Event>);
  static_assert(std::is_trivially_copyable_v<rund::telemetry::Finding>);
  static_assert(std::is_trivially_copyable_v<rund::telemetry::Findings>);
  static_assert(rund::telemetry::Findings::Capacity == 5u);
  constexpr rund::telemetry::Event telemetry{
      .source = rund::telemetry::Source::Compute,
      .level = rund::telemetry::Level::Basic,
      .session = 7u,
      .compute =
          {
              .backend = rund::compute::Backend::Cpu,
              .code = rund::compute::Code::Ok,
              .graph = 11u,
              .workers = 4u,
              .active_workers = 3u,
              .tiles = 8u,
              .dispatches = 2u,
              .buffer_allocations = 0u,
          },
  };
  static_assert(telemetry.source == rund::telemetry::Source::Compute &&
                telemetry.level == rund::telemetry::Level::Basic &&
                telemetry.session == 7u && telemetry.compute.graph == 11u &&
                telemetry.compute.backend == rund::compute::Backend::Cpu &&
                telemetry.compute.code == rund::compute::Code::Ok &&
                telemetry.compute.workers == 4u &&
                telemetry.compute.active_workers == 3u &&
                telemetry.compute.tiles == 8u &&
                telemetry.compute.dispatches == 2u &&
                telemetry.compute.buffer_allocations == 0u);
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
          "phase-total" &&
      telemetry.error().empty();
  auto inspect = [](const rund::telemetry::Event &) {};
  const rund::telemetry::Sink sink = rund::telemetry::bind(inspect);
  if (!telemetry_text || !sink ||
      sink.level() != rund::telemetry::Level::Basic) {
    return 2;
  }
  static_assert(noexcept(std::declval<const rund::Session::Status &>().ok()));
  static_assert(
      noexcept(std::declval<const rund::Session::Result &>().exit_code()));
  static_assert(std::is_same_v<decltype(rund::host::io::OpenOptions{}.mode),
                               std::uint32_t>);
  static_assert(
      noexcept(std::declval<const rund::host::io::ReadResult &>().ok()));
  static_assert(!std::is_copy_constructible_v<rund::host::io::Fd>);
  static_assert(std::is_nothrow_move_constructible_v<rund::host::io::Fd>);
  static_assert(std::is_trivially_copyable_v<rund::host::io::FdView>);
  static_assert(
      std::is_same_v<decltype(rund::net::NonblockingResult{}.native_error),
                     int>);
  static_assert(std::is_same_v<decltype(rund::net::NonblockingResult{}.code()),
                               rund::ReasonCode>);
  static_assert(!std::is_copy_constructible_v<rund::net::Socket>);
  static_assert(!std::is_copy_assignable_v<rund::net::Socket>);
  static_assert(std::is_nothrow_move_constructible_v<rund::net::Socket>);
  static_assert(std::is_trivially_copyable_v<rund::net::SocketView>);
  static_assert(sizeof(rund::net::SocketView) == 16u);
  static_assert(!std::is_copy_constructible_v<rund::net::ready::Ticket>);
  static_assert(std::is_nothrow_move_constructible_v<rund::net::ready::Ticket>);
  static_assert(!std::is_copy_constructible_v<rund::net::server::Task>);
  static_assert(std::is_nothrow_move_constructible_v<rund::net::server::Task>);
  static_assert(
      std::is_same_v<decltype(std::declval<rund::net::server::Task::Awaiter &>()
                                  .await_resume()),
                     rund::net::server::Result>);
  static_assert(
      noexcept(std::declval<const rund::net::ready::Ticket &>().error()));
  static_assert(
      std::is_same_v<decltype(rund::task::NetworkStats{}.bytes_received()),
                     std::uint64_t>);
  static_assert(
      std::is_same_v<decltype(rund::task::NetworkStats{}.bytes_sent()),
                     std::uint64_t>);
  const rund::net::Address address{};
  if (!address.bytes().empty()) {
    return 2;
  }

  const rund::PreparedMemory prepared_memory{
      .capacity =
          {
              .code = rund::ReasonCode::Ok,
              .worker_count = 1u,
              .requested_lane_capacity = 8u,
              .available_lane_capacity = 8u,
          },
      .lane_capacity = 8u,
      .lane_high_water = 3u,
  };
  bool memory_recorded = false;
  bool worker_recorded = true;
  rund::task::Status joined{};
  const rund::Session::Result run =
      rund::run(rund::SessionConfig{.workers = 1u}, [&] {
        memory_recorded = rund::record_memory(prepared_memory);
        const rund::task::Handle task = rund::task::spawn("sdk-work", [&] {
          worker_recorded = rund::record_memory(prepared_memory);
        });
        joined = rund::task::join(task);
      });
  if (!run) {
    return run.exit_code();
  }
  if (!joined) {
    return joined.exit_code();
  }
  if (!run.ok() || !run.error().empty() || !memory_recorded ||
      worker_recorded || !run.memory().capacity.ok() ||
      run.memory().lane_high_water != 3u) {
    return 2;
  }

  rund::SessionConfig config{.id = 2u, .workers = 1u};
  config.scheduler.task_workers = 1u;
  config.scheduler.task_capacity = 2u;
  config.scheduler.ready_queue_capacity = 2u;
  rund::Session session{};
  const rund::Session::Status opened = session.open(config);
  if (!opened) {
    return opened.exit_code();
  }

  const int operation = [&]() -> int {
    rund::task::Status first_join{};
    const rund::Session::Result first = session.scope([&] {
      first_join = rund::task::join(rund::task::spawn("first", [] {}));
    });
    if (!first) {
      return first.exit_code();
    }
    if (!first_join) {
      return first_join.exit_code();
    }
    if (first.tasks().spawned() != 1u ||
        first.tasks().task_record_allocations() != 1u) {
      return 2;
    }

    rund::task::Status second_join{};
    const rund::Session::Result second = session.scope([&] {
      second_join = rund::task::join(rund::task::spawn("second", [] {}));
    });
    if (!second) {
      return second.exit_code();
    }
    if (!second_join) {
      return second_join.exit_code();
    }
    if (second.tasks().spawned() != 1u ||
        second.tasks().task_record_allocations() != 0u ||
        second.tasks().task_record_reuses() != 1u) {
      return 2;
    }
    if (first.tasks().spawned() != 1u ||
        first.tasks().task_record_reuses() != 0u) {
      return 2;
    }
    return 0;
  }();

  const rund::Session::Status closed = session.close();
  if (operation != 0 && operation != 2) {
    return operation;
  }
  if (!closed) {
    return closed.exit_code();
  }
  return operation;
}
