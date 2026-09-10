#pragma once

#include "../registry/cache_model.hpp"
#include "../registry/graph.hpp"
#include "evidence.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::execution {

inline constexpr std::size_t BankCapacity = 2u;
inline constexpr std::size_t PredecessorCapacity = 2u;
inline constexpr std::size_t MutationCapacity = 2u;
inline constexpr std::uint32_t UseCapacity = 32u;
// A centered Window epoch may consume the K core pages plus one canonical
// neighbor on either side.  Slices stay bounded to three source pages per
// target because seal rejects a halo wider than one canonical page.
inline constexpr std::size_t WindowFootprintSourceCapacity = UseCapacity + 2u;
inline constexpr std::size_t WindowFootprintSliceCapacity = UseCapacity * 3u;

enum class Phase : std::uint8_t { Input, Dispatch, Output };
enum class Domain : std::uint8_t { HostService, Native };
enum class FetchFill : std::uint8_t {
  None,
  ZeroInactiveTail,
  RepeatBoundary,
  ConstantBoundary,
};

struct NodeId final {
  std::uint64_t epoch{};
  Phase phase{Phase::Input};

  [[nodiscard]] constexpr bool
  operator==(const NodeId &) const noexcept = default;
};

// One O(1) projection of the existing StreamPlan ready horizon. This carries
// no physical slot or victim decision: Authority owns those when it mints a
// Forecast ticket. `first_page`/`page_count` authenticate the same logical
// page run later consumed by the Promote/Dispatch node.
struct Forecast final {
  std::uint64_t epoch{};
  std::uint64_t first_page{};
  std::uint64_t page_count{};
  std::uint64_t prefetch_epoch{};
  std::uint64_t ready_epoch{};

  [[nodiscard]] constexpr bool
  operator==(const Forecast &) const noexcept = default;
};

// One exact logical materialization formula. GraphMaterialization remains the
// sole PageKey -> CacheKey projection. `logical_bytes` and `payload_bytes`
// define the page-local dirty cardinality; they do not mirror frame geometry.
// GraphMaterialization::boundary_extent is an invocation-wide semantic prefix
// identity, not a page-local byte count, and is therefore not bounded by
// either page_bytes or payload_bytes.
struct Materialization final {
  GraphMaterialization cache{};
  Access access{Access::Read};
  std::uint64_t logical_bytes{};
  std::uint64_t payload_bytes{};
  // Exact backing-to-frame projection for a read materialization. A Window
  // reads `read_prefix_bytes` before its core and `read_suffix_bytes` after
  // it; Scan keeps those zero and places its core at `target_prefix_bytes`.
  // Write materializations keep all three zero.
  std::uint64_t read_prefix_bytes{};
  std::uint64_t target_prefix_bytes{};
  std::uint64_t read_suffix_bytes{};
  // Pointwise may deterministically zero bytes outside the logical final
  // page because its inactive lanes cannot observe them. Window/halo fill is
  // a distinct scalar-aligned semantic recipe sealed as exact Clamp repeat or
  // Clip identity bytes.
  FetchFill fill{FetchFill::None};
  // Boundary recipes operate in complete scalar lanes. ConstantBoundary
  // repeats the low `fill_element_bytes` of `fill_value`; RepeatBoundary
  // copies the first/last scalar returned by the authenticated backing read.
  // None and ZeroInactiveTail keep both fields zero.
  std::uint64_t fill_element_bytes{};
  std::uint64_t fill_value{};
  std::uint64_t dirty_origin{};
  std::uint64_t next_use_base{NeverUse};
  std::uint64_t next_use_stride{};
  std::uint64_t retain_until{NeverUse};
};

// Authority-sealed byte range for one Direct backing read. `key` authenticates
// the canonical VSM page identity while offset/bytes and target_offset name
// the only legal backing mutation of the selected Host frame.
struct FetchSource final {
  CacheKey key{};
  std::uint64_t offset{};
  std::uint64_t bytes{};
  std::uint64_t target_offset{};
  std::uint64_t frame_bytes{};
  FetchFill fill{FetchFill::None};
  std::uint64_t fill_element_bytes{};
  std::uint64_t fill_value{};

  [[nodiscard]] constexpr bool
  operator==(const FetchSource &) const noexcept = default;
  [[nodiscard]] constexpr bool complete_frame() const noexcept {
    return target_offset == 0u && bytes == frame_bytes;
  }
  [[nodiscard]] constexpr bool materializes_frame() const noexcept {
    if (complete_frame()) {
      return true;
    }
    if (fill == FetchFill::ZeroInactiveTail) {
      return target_offset == 0u && bytes < frame_bytes &&
             fill_element_bytes == 0u && fill_value == 0u;
    }
    const bool scalar = fill_element_bytes == 1u || fill_element_bytes == 2u ||
                        fill_element_bytes == 4u || fill_element_bytes == 8u;
    return (fill == FetchFill::RepeatBoundary ||
            fill == FetchFill::ConstantBoundary) &&
           scalar && bytes >= fill_element_bytes &&
           target_offset <= frame_bytes &&
           bytes <= frame_bytes - target_offset &&
           target_offset % fill_element_bytes == 0u &&
           bytes % fill_element_bytes == 0u &&
           frame_bytes % fill_element_bytes == 0u;
  }
};

// Authority-sealed reuse of the overlap between two adjacent Direct input
// materializations. `key` names the prior resident page whose physical frame
// Authority must authenticate and pin. The copy slice plus the remaining
// backing slice reconstruct exactly the current FetchSource; neither the Host
// service nor the backend may recompute this geometry.
struct FetchReuseSource final {
  CacheKey key{};
  std::uint64_t source_offset{};
  std::uint64_t target_offset{};
  std::uint64_t bytes{};
  std::uint64_t read_offset{};
  std::uint64_t read_target_offset{};
  std::uint64_t read_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const FetchReuseSource &) const noexcept = default;
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return bytes != 0u;
  }
};

// One immutable canonical-page -> halo-expanded target-frame byte slice.
// `source` indexes WindowFootprintProjection::sources and `target` indexes the
// current epoch's K-wide Device input region.  Authority mints physical frames
// from these indices; callers cannot substitute a page, offset, or byte span.
struct WindowFootprintSlice final {
  std::uint32_t source{};
  std::uint32_t target{};
  std::uint64_t source_offset{};
  std::uint64_t target_offset{};
  std::uint64_t bytes{};

  [[nodiscard]] constexpr bool
  operator==(const WindowFootprintSlice &) const noexcept = default;
};

// Fixed-size O(K) projection of one centered Window epoch. `sources` are the
// unique canonical backing pages, `targets` are the exact expanded frame
// recipes already sealed by Direct Plan, and `slices` reconstruct every
// target from those pages without another geometry authority.
struct WindowFootprintProjection final {
  std::array<CacheUse, WindowFootprintSourceCapacity> sources{};
  std::array<FetchSource, UseCapacity> targets{};
  std::array<WindowFootprintSlice, WindowFootprintSliceCapacity> slices{};
  std::uint64_t epoch{};
  std::size_t source_count{};
  std::size_t target_count{};
  std::size_t slice_count{};
};

// Optional capability for one source-private transactional publication.
// `capability` and `generation` are minted together by the backing owner, not
// inferred by Authority. Both zero retains partial-output poison semantics;
// both nonzero seal external all-or-none publication into Plan identity.
struct Publication final {
  std::uint64_t backing{};
  std::uint64_t version{};
  std::uint64_t generation{};
  std::uint64_t capability{};
  DirtyExtent extent{};
};

// A service route names Authority-registered physical owners. HostService is
// runD Host work; Native is only prepared dispatch/progress. A route never
// claims that either endpoint is a GPU copy engine. Input Host regions may be
// wider than the K-wide Device bank: H>=K is the reusable Host cache ring,
// while each projected Node still authenticates at most K PageUse rows.
struct Route final {
  FrameRegion source{};
  FrameRegion target{};
};

// One O(1)-projected node with bounded exact PageUse materialization. Authority
// authenticates these keys, accesses, dirty extents, roles, and ranges; the
// backend may not reconstruct or replace them.
struct Node final {
  NodeId id{};
  Domain domain{Domain::HostService};
  std::uint32_t bank{};
  Route route{};
  std::array<CacheUse, UseCapacity> input{};
  std::array<CacheUse, UseCapacity> output{};
  std::size_t input_count{};
  std::size_t output_count{};
  std::array<FrameRegion, MutationCapacity> mutations{};
  std::size_t mutation_count{};
  std::uint32_t active_mask{};
  // Compatibility projection for the granular Host-service journal. Exact
  // invalidation consumes mutations[0..mutation_count), never this bool alone.
  bool may_write{};
};

enum class SealFailure : std::uint8_t {
  None,
  Invalid,
  Capacity,
  Overflow,
};

struct ReceiptSnapshot final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::uint64_t native_submissions{};
  Progress progress{};
  std::array<FailureEvidence, FailureCapacity> failures{};
  std::size_t failure_count{};
  std::uint64_t started_ns{};
  std::uint64_t completed_ns{};
};

} // namespace rund::compute::detail::residency::execution
