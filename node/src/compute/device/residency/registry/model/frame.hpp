#pragma once

#include "../../../../pipeline/residency/model.hpp"
#include "../cache_model.hpp"
#include "../../execution/registration/state.hpp"

#include <cstdint>
#include <memory>

namespace rund::compute::detail::residency::registry_model {

// One physical cache row.  Authority owns the only table of these records;
// this model header only names the row layout so private algorithms do not
// need to depend on the Authority declaration.
struct Frame final {
  CacheKey key{};
  std::uint64_t next_use{};
  std::uint64_t retain_until{NeverUse};
  FrameState state{FrameState::Empty};
  FrameTier tier{FrameTier::Device};
  FrameRole role{FrameRole::Input};
  DirtyExtent dirty{};
  // Zero identifies an ordinary non-aliased region. Nonzero values bind a
  // metadata row to one raw physical extent and one semantic page view over
  // that extent. Different views may have different frame counts/roles but
  // can never be simultaneously valid.
  std::uint64_t extent{};
  std::uint64_t view{};
  std::shared_ptr<const registration_detail::State> direct_registration{};
  // Registry mode keeps one table for every physical frame class. Released
  // regions remain reusable holes rather than a second free-frame ledger.
  bool assigned{};
  // Read-only cross-bank aliases pin this source independently from the
  // producer lease. Eviction and region release must wait for zero claims.
  std::uint32_t alias_claims{};
  // Nonzero only while one run-owned transaction has provisionally stamped
  // this physical output row. Empty/committed rows always carry zero.
  std::uint64_t transaction_generation{};
  // Nonzero only while one pooled view commit receipt owns this row.
  std::uint64_t view_commit_stamp{};
};

} // namespace rund::compute::detail::residency::registry_model
