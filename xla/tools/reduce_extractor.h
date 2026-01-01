#ifndef XLA_TOOLS_REDUCE_EXTRACTOR_H_
#define XLA_TOOLS_REDUCE_EXTRACTOR_H_

#include <memory>
#include <string>
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

struct ReduceExtractorConfig {
  std::string input_file{""};
  std::string input_format{""};
};

enum class RedFuncType {
  kUnknown,
  kSum,
  kProd,
  kMin,
  kMax,
  kAnd,
  kOr,
};

absl::string_view RedFuncString(RedFuncType func);

void ExtractReduceFunctions(std::unique_ptr<HloModule> module);

}  // namespace xla

#endif  // XLA_TOOLS_REDUCE_EXTRACTOR_H_
