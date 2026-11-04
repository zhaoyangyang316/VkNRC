#ifndef NN_KHR_GLSL
#define NN_KHR_GLSL

#ifndef NRC_SCALAR
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#endif
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_memory_scope_semantics : require

// Chosen from vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR. Unlike NV's
// fragment type, KHR distinguishes A, B and accumulator roles and dimensions.
#ifndef NRC_SCALAR
layout(constant_id = 1) const uint TILE_M = 16u;
layout(constant_id = 2) const uint TILE_N = 16u;
layout(constant_id = 3) const uint TILE_K = 16u;
#endif
#ifdef NRC_KHR_FP32
#define ACC_T float
#else
#define ACC_T float16_t
#endif
#define MAT_A coopmat<float16_t, gl_ScopeSubgroup, TILE_M, TILE_K, gl_MatrixUseA>
#define MAT_B coopmat<float16_t, gl_ScopeSubgroup, TILE_K, TILE_N, gl_MatrixUseB>
#define MAT_C coopmat<ACC_T, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator>

const uint WIDTH = 64u, SAMPLES = 16u, PLANE = SAMPLES * WIDTH;
layout(std430, binding = WEIGHTS_BINDING) readonly buffer KHRWeights { float16_t weights[]; };
#ifdef NN_BACKPROPAGATION
// Keep the existing float32 optimizer buffer format, using core uint atomics
// so the Android training pipeline does not require VK_EXT_shader_atomic_float.
layout(std430, binding = DWEIGHTS_BINDING) buffer KHRGradients { uint dw_bits[]; };
shared float16_t activations[6 * PLANE], derivatives[PLANE];
#ifdef NRC_SCALAR
// Scalar dW is reduced per weight in registers, leaving only a dX/forward plane.
shared ACC_T tile_result[PLANE];
#else
shared ACC_T tile_result[WIDTH * WIDTH];
#endif
#else
shared float16_t activations[2 * PLANE];
shared ACC_T tile_result[PLANE];
#endif

uint activationBase(uint layer) {
#ifdef NN_BACKPROPAGATION
	return layer * PLANE;
#else
	return (layer % 2u) * PLANE;
#endif
}

void forwardHidden(uint layer) {
	// W[out,in] * X[in,sample]. X and the result use column-major storage.
#ifdef NRC_SCALAR
	for (uint i = gl_LocalInvocationID.x; i < PLANE; i += WORKGROUP_SIZE) {
		uint x = activationBase(layer) + (i / WIDTH) * WIDTH;
		uint w = layer * WIDTH * WIDTH + (i % WIDTH) * WIDTH;
		float sum = 0.0;
		for (uint k = 0; k < WIDTH; ++k) sum += float(weights[w + k]) * float(activations[x + k]);
		tile_result[i] = ACC_T(sum);
	}
#else
	for (uint t = gl_SubgroupID; t < (WIDTH / TILE_M) * (SAMPLES / TILE_N); t += gl_NumSubgroups) {
		uint row = (t / (SAMPLES / TILE_N)) * TILE_M;
		uint col = (t % (SAMPLES / TILE_N)) * TILE_N;
		MAT_C sum = MAT_C(0);
		for (uint k = 0; k < WIDTH; k += TILE_K) {
			MAT_A w;
			MAT_B x;
			coopMatLoad(w, weights, layer * WIDTH * WIDTH + row * WIDTH + k, WIDTH, gl_CooperativeMatrixLayoutRowMajor);
			coopMatLoad(x, activations, activationBase(layer) + col * WIDTH + k, WIDTH, gl_CooperativeMatrixLayoutColumnMajor);
			sum = coopMatMulAdd(w, x, sum);
		}
		coopMatStore(sum, tile_result, col * WIDTH + row, WIDTH, gl_CooperativeMatrixLayoutColumnMajor);
	}
#endif
	barrier();
	for (uint i = gl_LocalInvocationID.x; i < PLANE; i += WORKGROUP_SIZE)
		activations[activationBase(layer + 1u) + i] = float16_t(max(float(tile_result[i]), 0.0));
	barrier();
}

// The final layer has only three stored output rows. Computing its small dot
// products directly avoids requiring a padded descriptor or reading beyond it.
vec3 forwardRGB(uint sample_id) {
	vec3 rgb = vec3(0);
	for (uint i = 0; i < WIDTH; ++i) {
		uint w = 5u * WIDTH * WIDTH + i;
		float x = float(activations[activationBase(5) + sample_id * WIDTH + i]);
		rgb += x * vec3(weights[w], weights[w + WIDTH], weights[w + 2u * WIDTH]);
	}
	return rgb;
}

#ifdef NN_BACKPROPAGATION
void addGradient(uint index, float value) {
	if (value == 0.0 || isnan(value) || isinf(value)) return;
	uint previous = atomicLoad(dw_bits[index], gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
	for (;;) {
		uint desired = floatBitsToUint(uintBitsToFloat(previous) + value);
		uint observed = atomicCompSwap(dw_bits[index], previous, desired, gl_ScopeQueueFamily,
		                               gl_StorageSemanticsBuffer, gl_SemanticsRelaxed,
		                               gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
		if (observed == previous) return;
		previous = observed;
	}
}

void backwardRGB() {
	for (uint i = gl_LocalInvocationID.x; i < 3u * WIDTH; i += WORKGROUP_SIZE) {
		float dw = 0.0;
		for (uint s = 0; s < SAMPLES; ++s)
			dw += float(derivatives[s * WIDTH + i / WIDTH]) * float(activations[5u * PLANE + s * WIDTH + i % WIDTH]);
		addGradient(5u * WIDTH * WIDTH + i, dw);
	}
	for (uint i = gl_LocalInvocationID.x; i < PLANE; i += WORKGROUP_SIZE) {
		uint w = 5u * WIDTH * WIDTH + i % WIDTH;
		uint d = (i / WIDTH) * WIDTH;
		float dx = dot(vec3(weights[w], weights[w + WIDTH], weights[w + 2u * WIDTH]),
		               vec3(derivatives[d], derivatives[d + 1], derivatives[d + 2]));
		tile_result[i] = ACC_T(activations[5u * PLANE + i] > 0.0 ? dx : 0.0);
	}
	barrier();
	for (uint i = gl_LocalInvocationID.x; i < PLANE; i += WORKGROUP_SIZE)
		derivatives[i] = float16_t(tile_result[i]);
	barrier();
}

void backwardHidden(uint layer) {
	// dW[out,in] = dY[out,sample] * X^T[sample,in].
#ifdef NRC_SCALAR
	for (uint i = gl_LocalInvocationID.x; i < WIDTH * WIDTH; i += WORKGROUP_SIZE) {
		float sum = 0.0;
		for (uint s = 0; s < SAMPLES; ++s)
			sum += float(derivatives[s * WIDTH + i / WIDTH]) * float(activations[layer * PLANE + s * WIDTH + i % WIDTH]);
		addGradient(layer * WIDTH * WIDTH + i, sum);
	}
#else
	for (uint t = gl_SubgroupID; t < (WIDTH / TILE_M) * (WIDTH / TILE_N); t += gl_NumSubgroups) {
		uint row = (t / (WIDTH / TILE_N)) * TILE_M;
		uint col = (t % (WIDTH / TILE_N)) * TILE_N;
		MAT_C sum = MAT_C(0);
		for (uint k = 0; k < SAMPLES; k += TILE_K) {
			MAT_A dy;
			MAT_B x;
			coopMatLoad(dy, derivatives, k * WIDTH + row, WIDTH, gl_CooperativeMatrixLayoutColumnMajor);
			coopMatLoad(x, activations, layer * PLANE + k * WIDTH + col, WIDTH, gl_CooperativeMatrixLayoutRowMajor);
			sum = coopMatMulAdd(dy, x, sum);
		}
		coopMatStore(sum, tile_result, row * WIDTH + col, WIDTH, gl_CooperativeMatrixLayoutRowMajor);
	}
	barrier();
	for (uint i = gl_LocalInvocationID.x; i < WIDTH * WIDTH; i += WORKGROUP_SIZE)
		addGradient(layer * WIDTH * WIDTH + i, float(tile_result[i]));
#endif
	barrier();
	if (layer == 0u) return;
	// dX[in,sample] = W^T[in,out] * dY[out,sample].
#ifdef NRC_SCALAR
	for (uint i = gl_LocalInvocationID.x; i < PLANE; i += WORKGROUP_SIZE) {
		uint d = (i / WIDTH) * WIDTH;
		uint w = layer * WIDTH * WIDTH + (i % WIDTH);
		float sum = 0.0;
		for (uint k = 0; k < WIDTH; ++k) sum += float(weights[w + k * WIDTH]) * float(derivatives[d + k]);
		tile_result[i] = ACC_T(sum);
	}
#else
	for (uint t = gl_SubgroupID; t < (WIDTH / TILE_M) * (SAMPLES / TILE_N); t += gl_NumSubgroups) {
		uint row = (t / (SAMPLES / TILE_N)) * TILE_M;
		uint col = (t % (SAMPLES / TILE_N)) * TILE_N;
		MAT_C sum = MAT_C(0);
		for (uint k = 0; k < WIDTH; k += TILE_K) {
			MAT_A w;
			MAT_B dy;
			coopMatLoad(w, weights, layer * WIDTH * WIDTH + k * WIDTH + row, WIDTH, gl_CooperativeMatrixLayoutColumnMajor);
			coopMatLoad(dy, derivatives, col * WIDTH + k, WIDTH, gl_CooperativeMatrixLayoutColumnMajor);
			sum = coopMatMulAdd(w, dy, sum);
		}
		coopMatStore(sum, tile_result, col * WIDTH + row, WIDTH, gl_CooperativeMatrixLayoutColumnMajor);
	}
#endif
	barrier();
	for (uint i = gl_LocalInvocationID.x; i < PLANE; i += WORKGROUP_SIZE)
		derivatives[i] = activations[layer * PLANE + i] > 0.0 ? float16_t(tile_result[i]) : float16_t(0);
	barrier();
}
#endif

// Preserve the existing dispatch/record ABI (128 examples/workgroup). Process
// groups of 16 examples so a full training forward tape fits in mobile shared memory.
vec3 evaluateKHR(in uvec4 input_words[8], vec3 target) {
	uint lane = gl_LocalInvocationID.x;
	uint sample_id = lane % SAMPLES;
	vec3 prediction = vec3(0);
	for (uint batch = 0; batch < WORKGROUP_SIZE / SAMPLES; ++batch) {
		bool own_sample = lane / SAMPLES == batch;
		if (own_sample) {
			for (uint i = 0; i < WIDTH; i += 2) {
				vec2 pair = unpackHalf2x16(input_words[i / 8u][(i % 8u) / 2u]);
				activations[sample_id * WIDTH + i] = float16_t(pair.x);
				activations[sample_id * WIDTH + i + 1u] = float16_t(pair.y);
			}
		}
		barrier();
		for (uint layer = 0; layer < 5; ++layer) forwardHidden(layer);
		if (own_sample) {
			prediction = forwardRGB(sample_id);
#ifdef NN_BACKPROPAGATION
			float luminance = dot(vec3(0.299, 0.587, 0.114), max(prediction, vec3(0)));
			vec3 d = 2.0 * LOSS_SCALE * (prediction - target) / (luminance * luminance + 0.01);
			for (uint c = 0; c < 3; ++c) derivatives[sample_id * WIDTH + c] = float16_t(d[c]);
#endif
		}
		barrier();
#ifdef NN_BACKPROPAGATION
		backwardRGB();
		for (int layer = 4; layer >= 0; --layer) backwardHidden(uint(layer));
#endif
	}
	return prediction;
}
#endif
