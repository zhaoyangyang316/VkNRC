#ifndef RNG_GLSL
#define RNG_GLSL

uint _RNG_state;

// Steps the RNG and returns a floating-point value in [0, 1).
float RNGNext() {
	// Condensed version of pcg_output_rxs_m_xs_32_32. Use exactly representable
	// 24-bit values so rounding cannot produce 1 (a zero-PDF cosine sample or
	// log(0) in the Beckmann sampler).
	_RNG_state = _RNG_state * 747796405 + 1;
	uint word = ((_RNG_state >> ((_RNG_state >> 28) + 4)) ^ _RNG_state) * 277803737;
	word = (word >> 22) ^ word;
	return float(word >> 8u) * (1.0 / 16777216.0);
}

void RNGSetState(in const uint rng_state) { _RNG_state = rng_state; }

#endif
