#pragma once

namespace rund::compute::detail::residency {
struct CacheUse;
}

namespace rund_node_test_pipeline_residency {

namespace authority_detail {

[[nodiscard]] const rund::compute::detail::residency::CacheUse &RegionA();

} // namespace authority_detail

[[nodiscard]] int CheckAuthorityCache();
[[nodiscard]] int CheckAuthorityMigration();
[[nodiscard]] int CheckAuthorityTransform();
[[nodiscard]] int CheckAuthorityCapacity();
[[nodiscard]] int CheckAuthorityLifetime();
[[nodiscard]] int CheckAuthorityGraph();
[[nodiscard]] int CheckAuthorityMigrationGraph();
[[nodiscard]] int CheckAuthorityRelocation();
[[nodiscard]] int CheckAuthorityViews();
[[nodiscard]] int CheckAuthorityViewCommit();
[[nodiscard]] int CheckAuthorityPrefetch();

} // namespace rund_node_test_pipeline_residency
