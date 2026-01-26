#include "xla/tools/reduce_extractor.h"

#include <iostream>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

namespace m = match;

std::pair<RedFuncType, PrimitiveType> TryToClassifyReductionFunction(
    const HloComputation* computation) {
  auto root = computation->root_instruction();

  if (Match(root, m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return {RedFuncType::kSum, root->shape().element_type()};
  } else if (Match(root,
                   m::MultiplyAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return {RedFuncType::kProd, root->shape().element_type()};
  } else if (Match(root,
                   m::MinimumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return {RedFuncType::kMin, root->shape().element_type()};
  } else if (Match(root,
                   m::MaximumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return {RedFuncType::kMax, root->shape().element_type()};
  } else if (Match(root, m::AndAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return {RedFuncType::kAnd, root->shape().element_type()};
  } else if (Match(root, m::OrAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return {RedFuncType::kOr, root->shape().element_type()};
  }

  std ::cout << "Could not classify reduction function:" << std::endl;
  std::cout << computation->ToString() << std::endl;
  return {RedFuncType::kUnknown, PRIMITIVE_TYPE_INVALID};
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
  absl::flat_hash_map<RedFuncType, absl::flat_hash_map<PrimitiveType, int>>
      func_type_freq;

  for (HloComputation* c : module->computations()) {
    for (HloInstruction* instr : c->instructions()) {
      if (instr->opcode() == HloOpcode::kReduce) {
        HloComputation* reduce_function = instr->to_apply();
        functions.insert(reduce_function);
        auto [func_type, primitive_type] =
            TryToClassifyReductionFunction(reduce_function);
        func_type_freq[func_type][primitive_type]++;
      }
    }
  }

  std::cout << "Total reduction functions found: " << functions.size()
            << std::endl;
  for (const auto& [func_type, freq_map] : func_type_freq) {
    for (const auto& [primitive_type, freq] : freq_map) {
      std::cout << "  " << RedFuncString(func_type) << " ("
                << primitive_util::LowercasePrimitiveTypeName(primitive_type)
                << "): " << freq << std::endl;
    }
  }
}

}  // namespace xla
