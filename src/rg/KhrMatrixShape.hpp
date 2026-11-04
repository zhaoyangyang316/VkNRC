#pragma once

#include <cstdint>

// Hidden layers are 64 wide. Each cooperative minibatch contains 16 samples;
// all three GEMMs (forward, activation gradient, weight gradient) must tile exactly.
constexpr bool SupportsKhrMatrixShape(uint32_t m, uint32_t n, uint32_t k) {
	return m && n && k && 64 % m == 0 && 16 % n == 0 && 16 % k == 0;
}

constexpr uint32_t KhrMatrixSharedBytes(bool training, bool fp32_accumulator) {
	const uint32_t half_bytes = 2, acc_bytes = fp32_accumulator ? 4 : 2;
	return training ? (6 + 1) * 16 * 64 * half_bytes + 64 * 64 * acc_bytes
	                : 2 * 16 * 64 * half_bytes + 16 * 64 * acc_bytes;
}
