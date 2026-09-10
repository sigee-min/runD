#pragma once

#include <rund/replay/record/base.hpp>

namespace rund::replay {

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
