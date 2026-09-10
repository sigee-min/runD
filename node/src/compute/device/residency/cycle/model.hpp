#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency::cycle {

// Three consecutive epochs are the smallest rolling window that exposes
// two-bank reuse. Input and Output are distinct physical owners, so H2D(e+2)
// may overlap D2H(e); only Dispatch(e+2) waits for epoch e's output-bank
// download. The storage bound is independent of dataset size.
inline constexpr std::size_t EpochCapacity = 3u;
inline constexpr std::uint32_t BankCapacity = 2u;
inline constexpr std::size_t PhaseCapacity = 3u;
inline constexpr std::size_t NodeCapacity = EpochCapacity * PhaseCapacity;
inline constexpr std::size_t EdgeCapacity = EpochCapacity * 2u + 1u;

enum class Domain : std::uint8_t { Host, Device, HostVisible };

enum class Phase : std::uint8_t { Upload, Dispatch, Download };

[[nodiscard]] constexpr std::size_t index(const Phase phase) noexcept {
  return static_cast<std::size_t>(phase);
}

// Each token names the Authority journal that owns the exact frames consumed
// by that phase. Two phases of one epoch may deliberately share one token;
// different epochs may not, because their terminal publication is distinct.
struct Epoch final {
  std::uint64_t ordinal{};
  std::uint32_t bank{};
  std::array<std::uint64_t, PhaseCapacity> tokens{};
  struct Range final {
    Domain domain{Domain::Host};
    std::uint32_t first{};
    std::uint32_t count{};
    std::uint32_t mask{};

    [[nodiscard]] constexpr bool
    operator==(const Range &) const noexcept = default;
  };
  std::array<Range, PhaseCapacity> reads{};
  std::array<Range, PhaseCapacity> writes{};
  std::array<bool, PhaseCapacity> may_write{};
};

struct Node final {
  std::uint64_t epoch{};
  std::uint64_t token{};
  std::uint32_t bank{};
  Phase phase{Phase::Upload};
  Domain source{Domain::Host};
  Domain target{Domain::Device};
  Epoch::Range read{};
  Epoch::Range write{};
  bool may_write{};
};

// Production Direct tranche: the live Authority execution token and the
// exact Input/Output frames touched by one dispatch. It intentionally does not
// impersonate an already-completed Host supply lease or a future drain lease.
struct Flight final {
  std::uint64_t ordinal{};
  std::uint64_t token{};
  std::uint32_t bank{};
  Epoch::Range input{};
  Epoch::Range output{};
  bool may_write{};
};

// Node indices name entries in Plan::nodes(). Edges are immediate ordering
// constraints; a backend may choose any topological execution that preserves
// them.
struct Edge final {
  std::uint8_t before{};
  std::uint8_t after{};
};

enum class Failure : std::uint8_t {
  None,
  Invalid,
  Capacity,
  Overflow,
};

struct Result;

class Plan final {
public:
  [[nodiscard]] std::span<const Node> nodes() const noexcept {
    return {nodes_.data(), node_count_};
  }
  [[nodiscard]] std::span<const Edge> edges() const noexcept {
    return {edges_.data(), edge_count_};
  }

private:
  friend Result seal(std::span<const Epoch>) noexcept;

  std::array<Node, NodeCapacity> nodes_{};
  std::array<Edge, EdgeCapacity> edges_{};
  std::size_t node_count_{};
  std::size_t edge_count_{};
};

struct Result final {
  Failure failure{Failure::Invalid};
  Plan plan{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == Failure::None;
  }
};

} // namespace rund::compute::detail::residency::cycle
