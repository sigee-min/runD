#pragma once

namespace program_compute_metadata_contract {

[[nodiscard]] int CheckResourceContracts();
[[nodiscard]] int CheckMetadataSurface();
[[nodiscard]] int CheckMetadataRejections();
[[nodiscard]] int CheckInputIdentity();
[[nodiscard]] int CheckRetention();

} // namespace program_compute_metadata_contract
