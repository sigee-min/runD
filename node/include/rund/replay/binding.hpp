#pragma once

#include <rund/replay/code.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

namespace rund::replay {

class Append;
class Checkpoint;
class Choice;
class Context;
class History;
class Record;
class Resume;
class Value;
class Writer;

enum class Restore : std::uint8_t {
  Restored,
  Failed,
};

struct Input final {
  std::uint64_t id = 0u;
  std::uint64_t schema = 0u;

  [[nodiscard]] friend constexpr bool operator==(const Input &,
                                                 const Input &) = default;
};

namespace detail {

struct SourceCall final {
  void *object = nullptr;
  std::uint64_t (*invoke)(void *, Writer &) = nullptr;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return object != nullptr && invoke != nullptr;
  }
};

struct RestoreCall final {
  void *object = nullptr;
  Restore (*invoke)(void *, std::span<const std::byte>) = nullptr;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return object != nullptr && invoke != nullptr;
  }
};

template <typename SourceFn>
std::uint64_t invoke_source(void *const raw, Writer &writer) {
  auto &source = *static_cast<SourceFn *>(raw);
  return std::invoke(source, writer);
}

template <typename SourceFn>
[[nodiscard]] SourceCall borrow_source(SourceFn &source) noexcept {
  return SourceCall{
      .object = const_cast<void *>(
          static_cast<const void *>(std::addressof(source))),
      .invoke = invoke_source<SourceFn>,
  };
}

template <typename RestoreFn>
Restore invoke_restore(void *const raw,
                       const std::span<const std::byte> bytes) {
  auto &restore = *static_cast<RestoreFn *>(raw);
  return std::invoke(restore, bytes);
}

template <typename RestoreFn>
[[nodiscard]] RestoreCall borrow_restore(RestoreFn &restore) noexcept {
  return RestoreCall{
      .object = const_cast<void *>(
          static_cast<const void *>(std::addressof(restore))),
      .invoke = invoke_restore<RestoreFn>,
  };
}

} // namespace detail

// A Channel binds one canonical input identity to one borrowed live source.
// Replay and Scenario never invoke that source: read() consumes the next
// canonical transcript row and returns its recorded sequence with its bytes.
class Channel final {
public:
  Channel(const Channel &) = delete;
  Channel &operator=(const Channel &) = delete;
  Channel(Channel &&) = delete;
  Channel &operator=(Channel &&) = delete;

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
  [[nodiscard]] constexpr Input input() const noexcept { return input_; }

  [[nodiscard]] Value read(Context &context) const;
  [[nodiscard]] Choice choice(std::uint64_t sequence,
                              std::span<const std::byte> bytes) const noexcept;

private:
  constexpr Channel(const Input input, const detail::SourceCall source,
                    const Code binding) noexcept
      : input_(input), source_(source), code_(binding) {
    if (code_ == Code::Ok && input_.id == 0u) {
      code_ = Code::InputIdInvalid;
    } else if (code_ == Code::Ok && input_.schema == 0u) {
      code_ = Code::InputSchemaInvalid;
    } else if (code_ == Code::Ok && !source_.valid()) {
      code_ = Code::InputInvalid;
    }
  }

  Input input_{};
  detail::SourceCall source_{};
  Code code_ = Code::InputInvalid;

  friend class Binding;
};

// Binding is the sole public state-schema and restore authority. It creates
// reusable input Channels and owns every checkpoint/history state boundary.
// Both callbacks are borrowed lvalues and are never copied or retained by the
// runtime beyond a synchronous call.
class Binding final {
public:
  Binding() noexcept = default;

  template <typename RestoreFn>
    requires std::is_object_v<RestoreFn> &&
             std::invocable<RestoreFn &, std::span<const std::byte>> &&
             std::same_as<
                 std::invoke_result_t<RestoreFn &,
                                      std::span<const std::byte>>,
                 Restore>
  Binding(std::uint64_t state_schema, RestoreFn &restore) noexcept
      : schema_(state_schema), restore_(detail::borrow_restore(restore)),
        checkpointable_(true) {}

  Binding(const Binding &) = delete;
  Binding &operator=(const Binding &) = delete;
  Binding(Binding &&) = delete;
  Binding &operator=(Binding &&) = delete;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code() == Code::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr Code code() const noexcept {
    return checkpointable_ && schema_ == 0u
               ? Code::StateSchemaInvalid
           : checkpointable_ && !restore_.valid()
               ? Code::CheckpointRestoreInvalid
               : Code::Ok;
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code());
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] constexpr std::uint64_t schema() const noexcept {
    return schema_;
  }
  [[nodiscard]] constexpr bool checkpointable() const noexcept {
    return checkpointable_ && code() == Code::Ok;
  }

  template <typename SourceFn>
    requires std::is_object_v<SourceFn> &&
             std::invocable<SourceFn &, Writer &> &&
             std::same_as<std::invoke_result_t<SourceFn &, Writer &>,
                          std::uint64_t>
  [[nodiscard]] Channel input(Input identity, SourceFn &source) const noexcept {
    return Channel{identity, detail::borrow_source(source), code()};
  }

  [[nodiscard]] Checkpoint
  checkpoint(const Record &completed, std::span<const std::byte> state) const
      noexcept;
  [[nodiscard]] Checkpoint
  advance(const Checkpoint &previous, const Record &completed,
          std::span<const std::byte> state) const noexcept;
  [[nodiscard]] Resume resume(const Checkpoint &checkpoint) const noexcept;
  [[nodiscard]] Append append(History &history, Record record,
                              std::span<const std::byte> state) const noexcept;

private:
  std::uint64_t schema_ = 0u;
  detail::RestoreCall restore_{};
  bool checkpointable_ = false;

  friend class Resume;
};

} // namespace rund::replay
