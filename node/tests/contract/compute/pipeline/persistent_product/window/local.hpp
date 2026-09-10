#pragma once

#include "../local.hpp"
#include "../model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_persistent_product::window_test {

inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t Radius = 2u;
inline constexpr std::size_t PayloadElements = FrameElements - Radius * 2u;

class WindowBacking final : public rund::compute::VirtualBacking {
public:
  explicit WindowBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override {
    return rund::compute::VirtualBackingTier::Persistent;
  }
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override {
    return 1u;
  }
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t, std::span<std::byte>) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t, std::span<const std::byte>) noexcept override;

  [[nodiscard]] bool seed(std::span<const std::byte>) noexcept;
  [[nodiscard]] bool observe(std::span<std::byte>) const noexcept;

private:
  std::vector<std::byte> bytes_;
};

struct WindowPrepared final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<WindowBacking> input{};
  std::shared_ptr<WindowBacking> output{};
  std::vector<std::uint32_t> expected{};
};

enum class Edge : std::uint8_t {
  Clamp,
  Clip,
};

[[nodiscard]] inline constexpr std::size_t
ProductPages(const std::uint64_t coordinates, const Edge edge) noexcept {
  static_cast<void>(edge);
  return static_cast<std::size_t>(coordinates * 2u - 1u);
}

[[nodiscard]] inline constexpr std::size_t
ProductElements(const std::uint64_t coordinates, const Edge edge) noexcept {
  const std::size_t pages = ProductPages(coordinates, edge);
  return pages * PayloadElements - 3u;
}

[[nodiscard]] bool PrepareWindowProduct(rund::compute::Backend, std::uint64_t,
                                        Edge, WindowPrepared &, bool &);
[[nodiscard]] bool CheckPersistentWindowFallback(
    rund::compute::Backend, NativeQueueCounter, std::uint64_t, Edge,
    bool &) noexcept;

[[nodiscard]] std::uint64_t BackingVersion(WindowBacking &) noexcept;
[[nodiscard]] std::uint64_t BackingRecovery(WindowBacking &) noexcept;
[[nodiscard]] bool ExactWindowOutput(const WindowPrepared &) noexcept;

} // namespace rund_node_test_persistent_product::window_test
