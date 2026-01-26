#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tools/reduce_extractor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace {
const char* const kUsage = R"(
This tool lets you extract all the reduction function in a given HloModule from a file (or stdin).

Usage:

  bazel run //xla/tools:reduce_extractor -- \
    --input_format=[hlo|mhlo|pb|pbtxt|stablehlo] \
    path/to/module
)";

}  // namespace

namespace xla {

namespace {

absl::Status RunReduceExtractor(const ReduceExtractorConfig& opts) {
  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(opts.input_file));
  }
  TF_ASSIGN_OR_RETURN(auto module, LoadModuleFromFile(opts.input_file, format));

  // TODO: Do we need to run the verifier here?
  HloVerifier verifier(
      HloVerifierOpts{}.WithLayoutSensitive(false).WithAllowMixedPrecision(
          true));
  TF_RETURN_IF_ERROR(verifier.Run(module.get()).status());

  ExtractReduceFunctions(std::move(module));

  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::ReduceExtractorConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &opts.input_format,
                "The format of the input file. Valid values:\n"
                "  hlo : HLO textual format\n"
                "  mhlo : MHLO in textual or bytecode format\n"
                "  pb : xla::HloProto in binary proto format\n"
                "  pbtxt : xla::HloProto in text proto format\n"
                "  stablehlo : StableHLO in textual or bytecode format")};
  // The usage string includes the message at the top of the file and the flags
  // defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    // Print the usage using cerr to avoid truncation by LOG.
    std::cerr << kUsageString;
    return 1;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  QCHECK(argc == 2) << "Must specify a single input file. Number of args: "
                    << argc;
  opts.input_file = argv[1];

  absl::Status status = xla::RunReduceExtractor(opts);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}
