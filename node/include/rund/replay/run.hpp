#pragma once

#include <rund/replay/state.hpp>
#include <rund/telemetry/event.hpp>

#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::replay::detail::scope {
class Timing;
}

namespace rund::replay {

class Live;
class Scenario;

namespace detail {

struct Call final {
  void *object = nullptr;
  void (*invoke)(void *, Context &, Session &) = nullptr;
};

struct Access final {
  [[nodiscard]] static Live live(Session &session, Call call);
  [[nodiscard]] static Record record(Session &session, std::uint64_t start,
                                     Call call);
  [[nodiscard]] static Check run(Session &session, const Record &expected,
                                 std::uint64_t start, Call call);
  [[nodiscard]] static Scenario scenario(Session &session,
                                         const Record &expected,
                                         std::span<const Choice> choices,
                                         std::uint64_t start, Call call);
  [[nodiscard]] static Record resume_record(Session &session,
                                            const Resume &resume, Call call);
  [[nodiscard]] static Check resume_run(Session &session, const Resume &resume,
                                        const Record &expected, Call call);
  [[nodiscard]] static Scenario resume_scenario(Session &session,
                                                const Resume &resume,
                                                const Record &expected,
                                                std::span<const Choice> choices,
                                                Call call);

private:
  [[nodiscard]] static telemetry::Event
  event(telemetry::Mode mode, telemetry::Preparation plan,
        std::uint64_t choices, const Session::Result &result) noexcept;
  [[nodiscard]] static telemetry::Event event(telemetry::Mode mode,
                                              telemetry::Preparation plan,
                                              std::uint64_t choices,
                                              Code code) noexcept;
  static void evidence(telemetry::Event &event, const Record &record) noexcept;
  static void emit(Session &session, telemetry::Event &&event) noexcept;
  [[nodiscard]] static Record
  capture(Session &session, std::uint64_t start, Call call,
          ::rund::replay::detail::scope::Timing &timing);
  [[nodiscard]] static Check
  replay(Session &session, const Record &expected, std::uint64_t start,
         Call call, std::shared_ptr<const void> prepared,
         telemetry::Preparation plan,
         ::rund::replay::detail::scope::Timing &timing);
  [[nodiscard]] static Scenario
  explore(Session &session, const Record &expected,
          std::span<const Choice> choices, std::uint64_t start, Call call,
          std::shared_ptr<const void> prepared,
          telemetry::Preparation preparation, ReplayScenarioPlan plan,
          ::rund::replay::detail::scope::Timing &timing);
  [[nodiscard]] static Session::Result
  execute(Session &session, telemetry::Mode mode,
          std::shared_ptr<const void> expected,
          std::shared_ptr<const void> choices, Call call, bool &callback_ran,
          ::rund::replay::detail::scope::Timing &timing);
  [[nodiscard]] static std::shared_ptr<const void>
  expected(Session &session, const Record &record, std::uint64_t start,
           Code &code, telemetry::Preparation &plan) noexcept;
  [[nodiscard]] static Record
  reject(Session &session, Code code,
         ::rund::replay::detail::scope::Timing &timing);
  [[nodiscard]] static Check
  reject(Session &session, const Record &expected, Code code,
         telemetry::Preparation plan,
         ::rund::replay::detail::scope::Timing &timing);
  [[nodiscard]] static Scenario
  reject(Session &session, Code code, telemetry::Preparation plan,
         std::uint64_t choices,
         ::rund::replay::detail::scope::Timing &timing);
};

} // namespace detail

class Live final {
public:
  Live(const Live &) = delete;
  Live &operator=(const Live &) = delete;
  Live(Live &&other) noexcept : result_(std::move(other.result_)) {
    other.result_ = Session::Result{};
  }
  Live &operator=(Live &&other) noexcept {
    if (this != &other) {
      result_ = std::move(other.result_);
      other.result_ = Session::Result{};
    }
    return *this;
  }

  [[nodiscard]] bool ok() const noexcept { return ::rund::replay::ok(code()); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code());
  }
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] std::uint64_t scope() const noexcept { return result_.scope(); }
  [[nodiscard]] const task::Stats &tasks() const noexcept {
    return result_.tasks();
  }
  [[nodiscard]] const ::rund::PreparedMemory &memory() const noexcept {
    return result_.memory();
  }
  [[nodiscard]] std::span<const task::Observation>
  observations() const noexcept {
    return result_.observations();
  }
  [[nodiscard]] std::span<const ::rund::host::Event> events() const noexcept {
    return result_.events();
  }
  [[nodiscard]] const ::rund::Trace &trace() const noexcept {
    return result_.trace();
  }
  [[nodiscard]] std::uint64_t trace_hash() const noexcept {
    return result_.trace_hash();
  }

private:
  explicit Live(Session::Result result) noexcept : result_(std::move(result)) {}

  Session::Result result_{};

  friend struct detail::Access;
};

class Scenario final {
public:
  Scenario(const Scenario &) noexcept = default;
  Scenario &operator=(const Scenario &) noexcept = default;
  Scenario(Scenario &&) noexcept = default;
  Scenario &operator=(Scenario &&) noexcept = default;

  [[nodiscard]] bool ok() const noexcept {
    return code_ == Code::Ok && actual_.has_value() && actual_->ok() &&
           diff_.has_value();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept {
    return ok()                ? Code::Ok
           : code_ == Code::Ok ? Code::ScenarioResultMovedFrom
                               : code_;
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code());
  }
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] bool callback_ran() const noexcept { return callback_ran_; }
  [[nodiscard]] bool matches() const noexcept { return ok() && diff_->ok(); }
  [[nodiscard]] const std::optional<Record> &actual() const noexcept {
    return actual_;
  }
  [[nodiscard]] const std::optional<Diff> &diff() const noexcept {
    return diff_;
  }

private:
  Scenario(Code code, std::optional<Record> actual, std::optional<Diff> diff,
           bool callback_ran) noexcept
      : code_(code), actual_(std::move(actual)), diff_(std::move(diff)),
        callback_ran_(callback_ran) {}

  Code code_ = Code::ScenarioNotRun;
  std::optional<Record> actual_{};
  std::optional<Diff> diff_{};
  bool callback_ran_ = false;

  friend Scenario detail::finish_scenario(const Record &, Session &,
                                          Session::Result &&, bool,
                                          std::uint64_t) noexcept;
  friend Scenario detail::fail_scenario(Code) noexcept;
};

namespace detail {

template <typename Callback>
void invoke(void *const raw, Context &context, Session &session) {
  auto &callback = *static_cast<Callback *>(raw);
  if constexpr (std::invocable<Callback &, Context &, Session &>) {
    static_assert(
        std::same_as<std::invoke_result_t<Callback &, Context &, Session &>,
                     void>,
        "replay callback must return void");
    std::invoke(callback, context, session);
  } else {
    static_assert(std::invocable<Callback &, Context &>,
                  "replay callback must accept Context& or Context&, Session&");
    static_assert(
        std::same_as<std::invoke_result_t<Callback &, Context &>, void>,
        "replay callback must return void");
    std::invoke(callback, context);
  }
}

template <typename Callback> [[nodiscard]] Call borrow(Callback &callback) {
  return Call{.object = const_cast<void *>(
                  static_cast<const void *>(std::addressof(callback))),
              .invoke = invoke<Callback>};
}

} // namespace detail

template <typename Callback>
[[nodiscard]] Live live(Session &session, Callback &&callback) {
  return detail::Access::live(session, detail::borrow(callback));
}

template <typename Callback>
[[nodiscard]] Record record(Session &session, Callback &&callback) {
  return detail::Access::record(session, detail::genesis_start(),
                                detail::borrow(callback));
}

template <typename Callback>
[[nodiscard]] Check run(Session &session, const Record &expected,
                        Callback &&callback) {
  return detail::Access::run(session, expected, detail::genesis_start(),
                             detail::borrow(callback));
}

template <typename Callback>
[[nodiscard]] Scenario scenario(Session &session, const Record &expected,
                                const std::span<const Choice> choices,
                                Callback &&callback) {
  return detail::Access::scenario(session, expected, choices,
                                  detail::genesis_start(),
                                  detail::borrow(callback));
}

template <typename Callback>
Record Resume::record(Session &session, Callback &&callback) const {
  return detail::Access::resume_record(session, *this,
                                       detail::borrow(callback));
}

template <typename Callback>
Check Resume::run(Session &session, const Record &expected,
                  Callback &&callback) const {
  return detail::Access::resume_run(session, *this, expected,
                                    detail::borrow(callback));
}

template <typename Callback>
Scenario Resume::scenario(Session &session, const Record &expected,
                          const std::span<const Choice> choices,
                          Callback &&callback) const {
  return detail::Access::resume_scenario(session, *this, expected, choices,
                                         detail::borrow(callback));
}

} // namespace rund::replay
