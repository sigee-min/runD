#pragma once

#include <rund/compute/buffer.hpp>
#include <rund/compute/abi/observe.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute {

namespace detail {
struct RunAccess;
}

class Run final {
public:
  Run(const Run &) noexcept;
  Run(Run &&) noexcept;
  Run &operator=(const Run &) noexcept;
  Run &operator=(Run &&) noexcept;
  ~Run();

  template <class T>
  [[nodiscard]] Status read(const Buffer<T>& buffer,
                            const std::span<T> output) const {
    if (buffer.size() != output.size()) {
      return Status::fail(Reason::ShapeMismatch);
    }
    return detail::read<T>(state(), buffer.state_, output);
  }

  [[nodiscard]] Stats stats() const noexcept {
    return detail::run_stats(state());
  }

private:
  friend struct detail::RunAccess;

  explicit Run(detail::RunState &&state) noexcept;

  [[nodiscard]] detail::RunState &state() noexcept;
  [[nodiscard]] const detail::RunState &state() const noexcept;

  static constexpr std::size_t StorageBytes = 1024u;
  alignas(std::uint64_t) std::array<std::byte, StorageBytes> storage_;
};

} // namespace rund::compute
