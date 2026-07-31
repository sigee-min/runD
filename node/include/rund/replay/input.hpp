#pragma once

#include <rund/replay/binding.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::telemetry {
enum class Mode : std::uint8_t;
}

namespace rund::replay {

class Value;

namespace detail {
struct Access;
}

class Choice final {
public:
  [[nodiscard]] constexpr const Input &input() const noexcept { return input_; }
  [[nodiscard]] constexpr std::uint64_t sequence() const noexcept {
    return sequence_;
  }
  [[nodiscard]] constexpr std::span<const std::byte> bytes() const noexcept {
    return bytes_;
  }

private:
  constexpr Choice(const Input input, const std::uint64_t sequence,
                   const std::span<const std::byte> bytes) noexcept
      : input_(input), sequence_(sequence), bytes_(bytes) {}

  Input input_{};
  std::uint64_t sequence_ = 0u;
  std::span<const std::byte> bytes_{};

  friend class Channel;
};

namespace detail {

struct Request final {
  Input input{};
  std::uint64_t sequence = 0u;
};

struct Lease final {
  const std::atomic<std::uint64_t> *generation = nullptr;
  std::uint64_t value = 0u;

  [[nodiscard]] bool valid() const noexcept {
    return generation != nullptr && value != 0u &&
           generation->load(std::memory_order_acquire) == value;
  }
};

struct ReplayInputCapture final {
  std::uint64_t token = 0u;
  Code code = Code::InputCaptureNotStarted;
  std::span<std::byte> bytes{};

  [[nodiscard]] bool ok() const noexcept { return ::rund::replay::ok(code); }
};

[[nodiscard]] ReplayInputCapture begin_input(Input input) noexcept;
void cancel_input(ReplayInputCapture capture) noexcept;
[[nodiscard]] Value finish_input(Request request, ReplayInputCapture capture,
                                 std::size_t byte_count, Lease lease);
[[nodiscard]] Value reject_input(ReplayInputCapture capture, Code code,
                                 Lease lease) noexcept;
void fail_input(Code code) noexcept;
[[nodiscard]] Value replay_input(Input input, Lease lease);

} // namespace detail

class Value final {
public:
  Value(const Value &) = delete;
  Value &operator=(const Value &) = delete;
  Value(Value &&) = delete;
  Value &operator=(Value &&) = delete;

  [[nodiscard]] bool ok() const noexcept { return code() == Code::Ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept {
    return lease_.valid() ? code_ : Code::ScopeExpired;
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code());
  }
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return ok() ? bytes_ : std::span<const std::byte>{};
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return ok() ? bytes_.size() : 0u;
  }
  [[nodiscard]] std::uint64_t sequence() const noexcept {
    return ok() ? sequence_ : 0u;
  }

private:
  Value(const std::span<const std::byte> bytes, const std::uint64_t sequence,
        const Code code, const detail::Lease lease) noexcept
      : bytes_(bytes), sequence_(sequence), code_(code), lease_(lease) {}

  std::span<const std::byte> bytes_{};
  std::uint64_t sequence_ = 0u;
  Code code_ = Code::InputNotResolved;
  detail::Lease lease_{};

  friend Value detail::finish_input(detail::Request, detail::ReplayInputCapture,
                                    std::size_t, detail::Lease);
  friend Value detail::reject_input(detail::ReplayInputCapture, Code,
                                    detail::Lease) noexcept;
  friend Value detail::replay_input(Input, detail::Lease);
  friend class Context;
};

class Writer final {
public:
  Writer(const Writer &) = delete;
  Writer &operator=(const Writer &) = delete;
  Writer(Writer &&) = delete;
  Writer &operator=(Writer &&) = delete;

  [[nodiscard]] bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return bytes_.size(); }
  [[nodiscard]] std::size_t remaining() const noexcept {
    return size_ <= bytes_.size() ? bytes_.size() - size_ : 0u;
  }

  [[nodiscard]] std::span<std::byte> acquire(const std::size_t count) noexcept {
    if (state_ != State::Open) {
      fail(Code::InputWriterStateInvalid);
      return {};
    }
    if (count > remaining()) {
      fail(Code::InputWriterCapacityExceeded);
      return {};
    }
    state_ = State::Acquired;
    pending_size_ = count;
    return bytes_.subspan(size_, count);
  }

  [[nodiscard]] bool commit(const std::size_t count) noexcept {
    if (state_ != State::Acquired || count > pending_size_) {
      fail(Code::InputWriterCommitInvalid);
      return false;
    }
    size_ += count;
    state_ = State::Open;
    pending_size_ = 0u;
    committed_ = true;
    return true;
  }

  [[nodiscard]] bool append(const std::span<const std::byte> source) noexcept {
    if (state_ != State::Open) {
      fail(Code::InputWriterStateInvalid);
      return false;
    }
    if (source.size() > remaining()) {
      fail(Code::InputWriterCapacityExceeded);
      return false;
    }
    std::copy(source.begin(), source.end(), bytes_.begin() + size_);
    size_ += source.size();
    committed_ = true;
    return true;
  }

private:
  enum class State : std::uint8_t { Open, Acquired, Failed };

  explicit Writer(const std::span<std::byte> bytes) noexcept : bytes_(bytes) {}

  void fail(const Code code) noexcept {
    if (state_ != State::Failed && code != Code::Ok) {
      code_ = code;
      state_ = State::Failed;
    }
  }

  [[nodiscard]] bool finish() noexcept {
    if ((!committed_ || state_ == State::Acquired) && state_ != State::Failed) {
      fail(Code::InputWriterUncommitted);
    }
    return ok();
  }

  std::span<std::byte> bytes_{};
  std::size_t size_ = 0u;
  std::size_t pending_size_ = 0u;
  Code code_ = Code::Ok;
  State state_ = State::Open;
  bool committed_ = false;

  friend class Context;
};

class Context final {
public:
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;

private:
  explicit Context(const telemetry::Mode mode,
                   const detail::Lease lease) noexcept
      : mode_(mode), lease_(lease) {}

  [[nodiscard]] Value read(Input input, detail::SourceCall source);

  [[nodiscard]] Value reject(Code code) const noexcept;

  telemetry::Mode mode_;
  detail::Lease lease_{};

  friend class Channel;
  friend struct detail::Access;
};

} // namespace rund::replay
