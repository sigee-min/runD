#pragma once

#include "store.hpp"

#include <cstdint>
#include <vector>

namespace rund::node::replay_detail::payload {

struct MaterializedRecord {
  Record metadata{};
  std::vector<std::byte> bytes{};
};

struct Materialization {
  std::vector<MaterializedRecord> records{};
  std::uint64_t payload_hash = 0u;
};

// Owns the supplied records. Rvalues therefore transfer payload storage;
// lvalues make the single copy required by this ownership boundary.
[[nodiscard]] Materialization
Materialize(std::vector<MaterializedRecord> records);

[[nodiscard]] Materialization Materialize(const Store &store);

[[nodiscard]] bool Equal(const Materialization &expected,
                         const Materialization &actual) noexcept;

[[nodiscard]] ::rund::node::replay_detail::payload::Archive
MakeArchive(const Materialization &materialized);

[[nodiscard]] BuildResult Build(const Materialization &materialized);

[[nodiscard]] BuildResult Build(const Materialization &materialized,
                                ::rund::replay::Storage storage);

} // namespace rund::node::replay_detail::payload
