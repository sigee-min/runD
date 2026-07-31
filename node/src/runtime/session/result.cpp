#include <rund/session.hpp>

#include <node/runtime/replay/host/archive.hpp>

#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace {
using Payload = rund::node::replay_detail::payload::Archive;
}

namespace rund {

Session::Result::Result() noexcept {
  static_assert(sizeof(Payload) <= payload_capacity);
  static_assert(alignof(Payload) <= alignof(std::max_align_t));
  static_assert(std::is_nothrow_default_constructible_v<Payload>);
  static_assert(std::is_nothrow_move_constructible_v<Payload>);
  static_assert(std::is_nothrow_move_assignable_v<Payload>);
  static_assert(std::is_nothrow_destructible_v<Payload>);
  std::construct_at(reinterpret_cast<Payload *>(payload_));
}

Session::Result::~Result() {
  std::destroy_at(static_cast<Payload *>(storage()));
}

Session::Result::Result(Result &&other) noexcept
    : code_(other.code_), telemetry_clock_reads_(other.telemetry_clock_reads_),
      scope_(other.scope_), tasks_(std::move(other.tasks_)),
      memory_(std::move(other.memory_)),
      observations_(std::move(other.observations_)),
      events_(std::move(other.events_)), input_rows_(other.input_rows_),
      input_bytes_(other.input_bytes_), ready_capacity_(other.ready_capacity_),
      trace_(std::move(other.trace_)), detail_(other.detail_) {
  std::construct_at(reinterpret_cast<Payload *>(payload_),
                    std::move(*static_cast<Payload *>(other.storage())));
}

Session::Result &Session::Result::operator=(Result &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  code_ = other.code_;
  telemetry_clock_reads_ = other.telemetry_clock_reads_;
  scope_ = other.scope_;
  tasks_ = std::move(other.tasks_);
  memory_ = std::move(other.memory_);
  observations_ = std::move(other.observations_);
  events_ = std::move(other.events_);
  *static_cast<Payload *>(storage()) =
      std::move(*static_cast<Payload *>(other.storage()));
  input_rows_ = other.input_rows_;
  input_bytes_ = other.input_bytes_;
  ready_capacity_ = other.ready_capacity_;
  trace_ = std::move(other.trace_);
  detail_ = other.detail_;
  return *this;
}

Session::Result::Result(
    const ::rund::ReasonCode code, const std::uint64_t scope, task::Stats tasks,
    PreparedMemory memory, std::vector<task::Observation> observations,
    std::vector<::rund::host::Event> events, const std::uint64_t input_rows,
    const std::uint64_t input_bytes, const std::uint64_t ready_capacity,
    Trace trace, const telemetry::Detail detail,
    const std::uint8_t telemetry_clock_reads) noexcept
    : code_(code), telemetry_clock_reads_(telemetry_clock_reads), scope_(scope),
      tasks_(std::move(tasks)), memory_(std::move(memory)),
      observations_(std::move(observations)), events_(std::move(events)),
      input_rows_(input_rows), input_bytes_(input_bytes),
      ready_capacity_(ready_capacity), trace_(std::move(trace)),
      detail_(detail) {
  std::construct_at(reinterpret_cast<Payload *>(payload_));
}

void *Session::Result::storage() noexcept {
  return std::launder(reinterpret_cast<Payload *>(payload_));
}

} // namespace rund
