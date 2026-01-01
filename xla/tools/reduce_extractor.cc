#include <iostream>

#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tools/reduce_extractor.h"

namespace xla {

namespace {

namespace m = match;

RedFuncType TryToClassifyReductionFunction(const HloComputation* computation) {
  auto root = computation->root_instruction();

  if (Match(root, m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return RedFuncType::kSum;
  } else if (Match(root,
                   m::MultiplyAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return RedFuncType::kProd;
  } else if (Match(root,
                   m::MinimumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return RedFuncType::kMin;
  } else if (Match(root,
                   m::MaximumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return RedFuncType::kMax;
  } else if (Match(root, m::AndAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return RedFuncType::kAnd;
  } else if (Match(root, m::OrAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return RedFuncType::kOr;
  }

  std ::cout << "Could not classify reduction function:" << std::endl;
  std::cout << computation->ToString() << std::endl;
  return RedFuncType::kUnknown;
}

}  // namespace

absl::string_view RedFuncString(RedFuncType func) {
  switch (func) {
    case RedFuncType::kUnknown:
      return "Unknown";
    case RedFuncType::kSum:
      return "Sum";
    case RedFuncType::kProd:
      return "Product";
    case RedFuncType::kMin:
      return "Minimum";
    case RedFuncType::kMax:
      return "Maximum";
    case RedFuncType::kAnd:
      return "And";
    case RedFuncType::kOr:
      return "Or";
  }
}

void ExtractReduceFunctions(std::unique_ptr<HloModule> module) {
  absl::flat_hash_set<HloComputation*> functions;
  absl::flat_hash_map<RedFuncType, int> func_type_freq;

  for (HloComputation* c : module->computations()) {
    for (HloInstruction* instr : c->instructions()) {
      if (instr->opcode() == HloOpcode::kReduce) {
        HloComputation* reduce_function = instr->to_apply();
        functions.insert(reduce_function);
        auto func_type = TryToClassifyReductionFunction(reduce_function);
        func_type_freq[func_type]++;
      }
    }
  }

  std::cout << "Total reduction functions found: " << functions.size()
            << std::endl;
  for (const auto& [func_type, freq] : func_type_freq) {
    std::cout << "  " << RedFuncString(func_type) << ": " << freq << std::endl;
  }
}

}  // namespace xla
