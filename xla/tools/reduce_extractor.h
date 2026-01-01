#ifndef XLA_TOOLS_REDUCE_EXTRACTOR_H_
#define XLA_TOOLS_REDUCE_EXTRACTOR_H_

#include <string>

namespace xla {

struct ReduceExtractorConfig {
  std::string input_file{""};
  std::string input_format{""};
};

}

#endif // XLA_TOOLS_REDUCE_EXTRACTOR_H_
