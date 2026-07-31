#pragma once

#include <rund/replay/input.hpp>
#include <rund/replay/limits.hpp>
#include <rund/replay/storage.hpp>
#include <rund/session.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::replay {

class Check;
class Checkpoint;
class Diff;
class Record;
class Save;
class Scenario;
class Window;
template <typename T> class Load;

namespace detail {

struct Access;

[[nodiscard]] Record build_record(Session &session, Session::Result &&result,
                                  std::uint64_t start_hash) noexcept;
[[nodiscard]] std::uint64_t genesis_start() noexcept;
[[nodiscard]] std::uint64_t
checkpoint_start(std::uint64_t checkpoint_hash) noexcept;
[[nodiscard]] Record fail_record(Code code) noexcept;
[[nodiscard]] Check check_result(const Record &expected, Session &session,
                                 Session::Result &&actual,
                                 std::uint64_t start_hash) noexcept;
[[nodiscard]] Check fail_check(const Record &expected, Code code) noexcept;
[[nodiscard]] Code ready_code(const Record &record) noexcept;

struct ReplayScenarioPlan final {
  std::shared_ptr<const void> choices{};
  Code code = Code::ScenarioNotPrepared;
  std::uint64_t bytes = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return ::rund::replay::ok(code) && choices != nullptr;
  }
};

struct Output final {
  void *state = nullptr;
  bool (*write)(void *, std::span<const std::byte>) noexcept = nullptr;

  [[nodiscard]] bool valid() const noexcept {
    return state != nullptr && write != nullptr;
  }
};

template <typename Sink> [[nodiscard]] Output output(Sink &sink) noexcept {
  using Owner = std::remove_reference_t<Sink>;
  return Output{
      .state =
          const_cast<void *>(static_cast<const void *>(std::addressof(sink))),
      .write =
          [](void *const state,
             const std::span<const std::byte> bytes) noexcept {
            try {
              return static_cast<bool>(
                  std::invoke(*static_cast<Owner *>(state), bytes));
            } catch (...) {
              return false;
            }
          },
  };
}

[[nodiscard]] ReplayScenarioPlan
prepare_scenario(Session &session, const Record &expected,
                 std::span<const Choice> choices) noexcept;
[[nodiscard]] Scenario finish_scenario(const Record &expected, Session &session,
                                       Session::Result &&actual,
                                       bool callback_ran,
                                       std::uint64_t start_hash) noexcept;
[[nodiscard]] Scenario fail_scenario(Code code) noexcept;

} // namespace detail

class Save final {
public:
  [[nodiscard]] constexpr bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::replay::exit_code(code_);
  }
  [[nodiscard]] constexpr std::uint64_t bytes() const noexcept {
    return bytes_;
  }
  [[nodiscard]] constexpr std::uint64_t writes() const noexcept {
    return writes_;
  }

private:
  constexpr Save(const Code code, const std::uint64_t bytes,
                 const std::uint64_t writes) noexcept
      : code_(code), bytes_(bytes), writes_(writes) {}

  Code code_ = Code::ArtifactNotSaved;
  std::uint64_t bytes_ = 0u;
  std::uint64_t writes_ = 0u;

  friend class Record;
  friend class Checkpoint;
};

struct Mismatch {
  Code code = Code::OutcomeMismatch;
  std::string_view field{};
  std::uint64_t expected = 0u;
  std::uint64_t actual = 0u;
};

struct InputPoint final {
  std::size_t index = 0u;
  Input input{};
  std::uint64_t sequence = 0u;
  std::uint64_t size = 0u;
  std::uint64_t hash = 0u;

  [[nodiscard]] friend constexpr bool operator==(const InputPoint &,
                                                 const InputPoint &) = default;
};

struct Capture final {
  std::uint64_t sequence = 0u;
  ::rund::host::EventKind kind = ::rund::host::EventKind::None;
  std::span<const std::byte> bytes{};
  std::uint64_t hash = 0u;
};

struct Trace {
  std::uint64_t event = 0u;
  ::rund::TraceCode code{};
  std::uint64_t snapshot_state = 0u;
  std::uint32_t snapshot_active_compute_jobs = 0u;
  bool snapshot_scope_active = false;
  ::rund::ReasonCode snapshot_code = ::rund::ReasonCode::Ok;
  std::uint64_t sequence = 0u;

  [[nodiscard]] std::string_view error() const noexcept { return code.error(); }
  [[nodiscard]] std::string_view snapshot_error() const noexcept {
    return snapshot_code == ::rund::ReasonCode::Ok
               ? std::string_view{}
               : ::rund::ReasonString(snapshot_code);
  }
};

// Replay evidence is immutable after construction. Facade copies share the
// same immutable evidence owner; callers never manage native trace pointers or
// hash rebinding.
class Record final {
public:
  Record(const Record &) noexcept = default;
  Record &operator=(const Record &) noexcept = default;
  Record(Record &&other) noexcept
      : data_(std::move(other.data_)), code_(other.code_) {
    other.code_ = Code::RecordMovedFrom;
  }
  Record &operator=(Record &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      code_ = other.code_;
      other.code_ = Code::RecordMovedFrom;
    }
    return *this;
  }

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] int exit_code() const noexcept;
  [[nodiscard]] const task::Stats &tasks() const noexcept;
  [[nodiscard]] std::size_t observation_count() const noexcept;
  [[nodiscard]] std::size_t host_event_count() const noexcept;
  [[nodiscard]] std::size_t input_count() const noexcept;
  [[nodiscard]] std::uint64_t input_hash() const noexcept;
  [[nodiscard]] std::span<const Capture> captures() const noexcept;
  [[nodiscard]] std::uint64_t capture_hash() const noexcept;
  [[nodiscard]] ::rund::replay::DiagnosticReport
  capture_report() const noexcept;
  [[nodiscard]] ::rund::replay::StorageReport storage_report() const noexcept;
  [[nodiscard]] std::size_t trace_record_count() const noexcept;
  [[nodiscard]] std::uint64_t semantic_hash() const noexcept;
  [[nodiscard]] std::uint64_t operation_hash() const noexcept;
  [[nodiscard]] std::uint64_t observation_hash() const noexcept;
  [[nodiscard]] std::uint64_t host_event_hash() const noexcept;
  [[nodiscard]] std::uint64_t transcript_hash() const noexcept;
  [[nodiscard]] std::uint64_t trace_hash() const noexcept;
  [[nodiscard]] std::uint64_t start_hash() const noexcept;
  [[nodiscard]] std::uint64_t hash() const noexcept;
  template <typename Sink>
    requires std::invocable<Sink &, std::span<const std::byte>> &&
             std::convertible_to<
                 std::invoke_result_t<Sink &, std::span<const std::byte>>, bool>
  [[nodiscard]] Save save(Sink &&sink) const noexcept {
    return persist(detail::output(sink));
  }
  [[nodiscard]] static Load<Record> load(std::span<const std::byte> artifact,
                                         Limits limits = {}) noexcept;

private:
  struct Data;

  explicit Record(std::shared_ptr<const Data> data) noexcept
      : data_(std::move(data)) {}
  explicit Record(const Code code) noexcept : code_(code) {}
  [[nodiscard]] Save persist(detail::Output output) const noexcept;

  std::shared_ptr<const Data> data_{};
  Code code_ = Code::RecordMovedFrom;

  friend Record detail::build_record(Session &, Session::Result &&,
                                     std::uint64_t) noexcept;
  friend Record detail::fail_record(Code) noexcept;
  friend Code detail::ready_code(const Record &) noexcept;
  friend Check detail::check_result(const Record &, Session &,
                                    Session::Result &&, std::uint64_t) noexcept;
  friend Check check(const Record &, const Record &) noexcept;
  friend Diff diff(const Record &, const Record &) noexcept;
  friend Window window(const Record &, const Record &, std::size_t) noexcept;
  friend detail::ReplayScenarioPlan
  detail::prepare_scenario(Session &, const Record &,
                           std::span<const Choice>) noexcept;
  friend class Binding;

  friend struct detail::Access;
};

class Check final {
public:
  Check(const Check &) noexcept = default;
  Check &operator=(const Check &) noexcept = default;
  Check(Check &&other) noexcept
      : code_(other.code_), expected_hash_(other.expected_hash_),
        actual_(std::move(other.actual_)) {
    other.reset_moved_from();
  }
  Check &operator=(Check &&other) noexcept {
    if (this != &other) {
      code_ = other.code_;
      expected_hash_ = other.expected_hash_;
      actual_ = std::move(other.actual_);
      other.reset_moved_from();
    }
    return *this;
  }

  [[nodiscard]] constexpr bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] constexpr std::uint64_t expected_hash() const noexcept {
    return expected_hash_;
  }
  [[nodiscard]] std::uint64_t actual_hash() const noexcept {
    return actual_.has_value() ? actual_->hash() : 0u;
  }
  [[nodiscard]] const std::optional<Record> &actual() const noexcept {
    return actual_;
  }

private:
  Check(const Code code, const std::uint64_t expected_hash,
        std::optional<Record> actual) noexcept
      : code_(code), expected_hash_(expected_hash), actual_(std::move(actual)) {
  }

  void reset_moved_from() noexcept {
    code_ = Code::CheckMovedFrom;
    expected_hash_ = 0u;
    actual_.reset();
  }

  Code code_ = Code::NotChecked;
  std::uint64_t expected_hash_ = 0u;
  std::optional<Record> actual_{};

  friend Check detail::check_result(const Record &, Session &,
                                    Session::Result &&, std::uint64_t) noexcept;
  friend Check detail::fail_check(const Record &, Code) noexcept;
  friend Check check(const Record &, const Record &) noexcept;
};

class Diff final {
public:
  Diff(const Diff &) noexcept = default;
  Diff &operator=(const Diff &) noexcept = default;
  Diff(Diff &&other) noexcept
      : data_(std::move(other.data_)), code_(other.code_) {
    other.code_ = Code::DiffMovedFrom;
  }
  Diff &operator=(Diff &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      code_ = other.code_;
      other.code_ = Code::DiffMovedFrom;
    }
    return *this;
  }

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] std::size_t mismatch_count() const noexcept;
  [[nodiscard]] std::optional<Mismatch>
  mismatch(std::size_t index) const noexcept;

private:
  struct Data;

  explicit Diff(std::shared_ptr<const Data> data) noexcept
      : data_(std::move(data)) {}
  explicit Diff(const Code code) noexcept : code_(code) {}

  std::shared_ptr<const Data> data_{};
  Code code_ = Code::DiffMovedFrom;

  friend Diff diff(const Record &, const Record &) noexcept;
};

class Window final {
public:
  Window(const Window &) noexcept = default;
  Window &operator=(const Window &) noexcept = default;
  Window(Window &&other) noexcept
      : data_(std::move(other.data_)), code_(other.code_) {
    other.code_ = Code::WindowMovedFrom;
  }
  Window &operator=(Window &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      code_ = other.code_;
      other.code_ = Code::WindowMovedFrom;
    }
    return *this;
  }

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }

  [[nodiscard]] std::optional<std::size_t> observation_index() const noexcept;
  [[nodiscard]] std::span<const task::Observation>
  expected_observations() const noexcept;
  [[nodiscard]] std::span<const task::Observation>
  actual_observations() const noexcept;

  [[nodiscard]] std::optional<std::size_t> host_event_index() const noexcept;
  [[nodiscard]] std::span<const ::rund::host::Event>
  expected_host_events() const noexcept;
  [[nodiscard]] std::span<const ::rund::host::Event>
  actual_host_events() const noexcept;

  [[nodiscard]] std::optional<std::size_t> input_index() const noexcept;
  [[nodiscard]] std::span<const InputPoint> expected_inputs() const noexcept;
  [[nodiscard]] std::span<const InputPoint> actual_inputs() const noexcept;

  [[nodiscard]] std::optional<std::size_t> trace_record_index() const noexcept;
  [[nodiscard]] std::span<const Trace> expected_trace() const noexcept;
  [[nodiscard]] std::span<const Trace> actual_trace() const noexcept;

private:
  struct Data;

  explicit Window(std::shared_ptr<const Data> data) noexcept
      : data_(std::move(data)) {}
  explicit Window(const Code code) noexcept : code_(code) {}

  std::shared_ptr<const Data> data_{};
  Code code_ = Code::WindowMovedFrom;

  friend Window window(const Record &, const Record &, std::size_t) noexcept;
};

[[nodiscard]] Check check(const Record &expected,
                          const Record &actual) noexcept;
[[nodiscard]] Diff diff(const Record &expected, const Record &actual) noexcept;
[[nodiscard]] Window window(const Record &expected, const Record &actual,
                            std::size_t context = 2u) noexcept;

} // namespace rund::replay
