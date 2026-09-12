#pragma once
#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

namespace rund_node_test_virtual::product::graph_forecast_window {
inline constexpr std::size_t Elements = 73u;
inline constexpr std::size_t PageElements = 16u;
inline constexpr std::size_t Inputs = 4u;
inline constexpr std::size_t Stages = 5u;
inline constexpr std::size_t Leaves = 340u;
using Program = rund::compute::Program<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t)>;
struct Control final {
  std::mutex gate;
  std::condition_variable changed;
  std::size_t active{};
  std::size_t peak{};
  bool slow_started{};
  bool refill_started{};
  bool refilled_while_slow{};
  bool timeout{};
  bool fail{};
  void reset(bool failure) noexcept;
  [[nodiscard]] bool complete() noexcept;
};
class Backing final : public rund::compute::VirtualBacking {
public:
  Backing(std::shared_ptr<Control>, std::size_t input);
  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t, std::span<std::byte>) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t, std::span<const std::byte>) noexcept override;
  std::array<std::uint64_t, Elements> values{};

private:
  std::shared_ptr<Control> control_;
  std::size_t input_{};
};
[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &);
[[nodiscard]] std::uint64_t expected(std::size_t) noexcept;
} // namespace rund_node_test_virtual::product::graph_forecast_window
