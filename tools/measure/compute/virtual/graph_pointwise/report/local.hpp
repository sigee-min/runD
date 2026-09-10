#pragma once

namespace rund::measure::compute::virtual_graph_pointwise {

struct Facts;
struct Result;

void WriteCsvRow(const Result &result, const char *kind, const Facts &facts,
                 bool timing);
void WriteDiagnostics(const Result &result);

} // namespace rund::measure::compute::virtual_graph_pointwise
