#include "core_functions/scalar/string_functions.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar/string_common.hpp"
#include "utf8proc.hpp"

#include <string.h>

namespace duckdb {

template <bool LTRIM, bool RTRIM>
struct TrimOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, StringHeap &heap) {
		auto data = input.GetData();
		auto size = input.GetSize();

		int32_t codepoint;

		// Find the first character that is not left trimmed
		idx_t begin = 0;
		if (LTRIM) {
			while (begin < size) {
				auto bytes = DecodeCodepoint(data + begin, size - begin, codepoint);
				if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
					break;
				}
				begin += bytes;
			}
		}

		// Find the last character that is not right trimmed
		idx_t end;
		if (RTRIM) {
			end = begin;
			for (auto next = begin; next < size;) {
				next += DecodeCodepoint(data + next, size - next, codepoint);
				if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
					end = next;
				}
			}
		} else {
			end = size;
		}

		// Copy the trimmed string
		auto target = heap.EmptyString(end - begin);
		auto output = target.GetDataWriteable();
		memcpy(output, data + begin, end - begin);

		target.Finalize();
		return target;
	}
};

template <bool LTRIM, bool RTRIM>
static void UnaryTrimFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, TrimOperator<LTRIM, RTRIM>>(args.data[0], result);
}

//! The set of codepoints to trim; ASCII codepoints are kept in a bitmap so a lookup is a shift and a mask
struct TrimCodepointSet {
	void Initialize(string_t characters) {
		ascii[0] = 0;
		ascii[1] = 0;
		non_ascii.clear();
		auto data = characters.GetData();
		auto size = characters.GetSize();
		for (idx_t pos = 0; pos < size;) {
			int32_t codepoint;
			pos += DecodeCodepoint(data + pos, size - pos, codepoint);
			if (codepoint >= 0 && codepoint < 128) {
				ascii[codepoint >> 6] |= 1ULL << (codepoint & 63);
			} else {
				non_ascii.push_back(codepoint);
			}
		}
		std::sort(non_ascii.begin(), non_ascii.end());
	}

	bool ContainsByte(char c) const {
		auto byte = static_cast<uint8_t>(c);
		return byte < 128 && (ascii[byte >> 6] >> (byte & 63)) & 1;
	}

	bool Contains(int32_t codepoint) const {
		if (codepoint >= 0 && codepoint < 128) {
			return (ascii[codepoint >> 6] >> (codepoint & 63)) & 1;
		}
		return std::binary_search(non_ascii.begin(), non_ascii.end(), codepoint);
	}

	//! Bit c is set if ASCII codepoint c is trimmed: codepoints 0-63 in ascii[0], 64-127 in ascii[1]
	uint64_t ascii[2];
	//! Trimmed non-ASCII codepoints, sorted for binary search
	vector<int32_t> non_ascii;
};

//! Returns a slice of the input without copying it, so the input's string data must outlive the result
template <bool LTRIM, bool RTRIM>
static string_t TrimCodepoints(string_t input, const TrimCodepointSet &trim_set) {
	auto data = input.GetData();
	auto size = input.GetSize();

	idx_t begin = 0;
	idx_t end = size;
	if (trim_set.non_ascii.empty()) {
		// bytes of multi-byte characters are never ASCII, so the bytes can be scanned directly
		if (LTRIM) {
			while (begin < end && trim_set.ContainsByte(data[begin])) {
				begin++;
			}
		}
		if (RTRIM) {
			while (end > begin && trim_set.ContainsByte(data[end - 1])) {
				end--;
			}
		}
		return string_t(data + begin, UnsafeNumericCast<uint32_t>(end - begin));
	}

	int32_t codepoint;

	// Find the first character that is not left trimmed
	if (LTRIM) {
		while (begin < size) {
			auto bytes = DecodeCodepoint(data + begin, size - begin, codepoint);
			if (!trim_set.Contains(codepoint)) {
				break;
			}
			begin += bytes;
		}
	}

	// Find the last character that is not right trimmed
	if (RTRIM) {
		end = begin;
		for (auto next = begin; next < size;) {
			next += DecodeCodepoint(data + next, size - next, codepoint);
			if (!trim_set.Contains(codepoint)) {
				end = next;
			}
		}
	}
	return string_t(data + begin, UnsafeNumericCast<uint32_t>(end - begin));
}

template <bool LTRIM, bool RTRIM>
static void BinaryTrimFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &strings = input.data[0];
	auto &characters_vector = input.data[1];
	TrimCodepointSet trim_set;
	if (characters_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// a NULL trim set makes every result NULL
		if (ConstantVector::IsNull(characters_vector)) {
			ConstantVector::SetNull(result, count_t(input.size()));
			return;
		}
		trim_set.Initialize(ConstantVector::GetData<string_t>(characters_vector)[0]);
		UnaryExecutor::Execute<string_t, string_t>(
		    strings, result, [&](string_t str) { return TrimCodepoints<LTRIM, RTRIM>(str, trim_set); });
	} else {
		BinaryExecutor::Execute<string_t, string_t, string_t>(strings, characters_vector, result,
		                                                      [&](string_t str, string_t characters) {
			                                                      trim_set.Initialize(characters);
			                                                      return TrimCodepoints<LTRIM, RTRIM>(str, trim_set);
		                                                      });
	}
	// the results point into the input strings, so the result vector keeps their buffers alive
	StringVector::AddHeapReference(result, strings);
}

ScalarFunctionSet TrimFun::GetFunctions() {
	ScalarFunctionSet trim;
	ScalarFunction unary({}, LogicalType::VARCHAR, UnaryTrimFunction<true, true>);
	unary.GetSignature().AddParameter("string", LogicalType::VARCHAR);
	trim.AddFunction(unary);

	ScalarFunction binary({}, LogicalType::VARCHAR, BinaryTrimFunction<true, true>);
	binary.GetSignature().AddParameter("string", LogicalType::VARCHAR).AddParameter("characters", LogicalType::VARCHAR);
	trim.AddFunction(binary);
	return trim;
}

ScalarFunctionSet LtrimFun::GetFunctions() {
	ScalarFunctionSet ltrim;
	ScalarFunction unary({}, LogicalType::VARCHAR, UnaryTrimFunction<true, false>);
	unary.GetSignature().AddParameter("string", LogicalType::VARCHAR);
	ltrim.AddFunction(unary);

	ScalarFunction binary({}, LogicalType::VARCHAR, BinaryTrimFunction<true, false>);
	binary.GetSignature().AddParameter("string", LogicalType::VARCHAR).AddParameter("characters", LogicalType::VARCHAR);
	ltrim.AddFunction(binary);
	return ltrim;
}

ScalarFunctionSet RtrimFun::GetFunctions() {
	ScalarFunctionSet rtrim;
	ScalarFunction unary({}, LogicalType::VARCHAR, UnaryTrimFunction<false, true>);
	unary.GetSignature().AddParameter("string", LogicalType::VARCHAR);
	rtrim.AddFunction(unary);

	ScalarFunction binary({}, LogicalType::VARCHAR, BinaryTrimFunction<false, true>);
	binary.GetSignature().AddParameter("string", LogicalType::VARCHAR).AddParameter("characters", LogicalType::VARCHAR);
	rtrim.AddFunction(binary);
	return rtrim;
}

} // namespace duckdb
