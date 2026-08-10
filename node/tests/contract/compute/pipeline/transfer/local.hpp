#pragma once

namespace rund_node_test_pipeline_transfer {

[[nodiscard]] bool CheckBatchSuccessAndAccounting();
[[nodiscard]] bool CheckBatchClaimAndFailure();
[[nodiscard]] bool CheckBatchDownloadAndHash();

} // namespace rund_node_test_pipeline_transfer
