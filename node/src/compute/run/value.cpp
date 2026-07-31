#include "state.hpp"

#include "../host.hpp"

#include <rund/compute/run.hpp>

#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute {

Run::Run(detail::RunState &&value) noexcept {
  static_assert(sizeof(detail::RunState) <= sizeof(storage_));
  static_assert(alignof(detail::RunState) <= alignof(Run));
  static_assert(std::is_nothrow_move_constructible_v<detail::RunState>);
  std::construct_at(reinterpret_cast<detail::RunState *>(storage_.data()),
                    std::move(value));
}

Run::Run(const Run &other) noexcept {
  static_assert(std::is_nothrow_copy_constructible_v<detail::RunState>);
  std::construct_at(reinterpret_cast<detail::RunState *>(storage_.data()),
                    other.state());
}

Run::Run(Run &&other) noexcept {
  static_assert(std::is_nothrow_move_constructible_v<detail::RunState>);
  std::construct_at(reinterpret_cast<detail::RunState *>(storage_.data()),
                    std::move(other.state()));
}

Run &Run::operator=(const Run &other) noexcept {
  static_assert(std::is_nothrow_copy_assignable_v<detail::RunState>);
  if (this != &other) {
    state() = other.state();
  }
  return *this;
}

Run &Run::operator=(Run &&other) noexcept {
  static_assert(std::is_nothrow_move_assignable_v<detail::RunState>);
  if (this != &other) {
    state() = std::move(other.state());
  }
  return *this;
}

Run::~Run() { std::destroy_at(&state()); }

detail::RunState &Run::state() noexcept {
  return *std::launder(reinterpret_cast<detail::RunState *>(storage_.data()));
}

const detail::RunState &Run::state() const noexcept {
  return *std::launder(
      reinterpret_cast<const detail::RunState *>(storage_.data()));
}

namespace detail {

struct RunAccess final {
  [[nodiscard]] static Run make(RunState &&state) noexcept {
    return Run{std::move(state)};
  }
};

Result<Run>
execute_buffers(const std::shared_ptr<ProgramState> &program,
                const std::span<const std::shared_ptr<BufferState>> inputs,
                const std::span<const std::shared_ptr<BufferState>> outputs) {
  auto result = run_buffers(program, inputs, outputs);
  if (!result) {
    return Result<Run>::fail(result.reason());
  }
  return Result<Run>::success(RunAccess::make(std::move(result).value()));
}

} // namespace detail
} // namespace rund::compute
