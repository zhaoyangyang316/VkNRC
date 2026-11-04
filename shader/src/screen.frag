#version 450

layout(input_attachment_index = 0, binding = 0) uniform subpassInput uColor;
layout(binding = 1, rgba32f) uniform image2D uAccumulate;
layout(binding = 2, rgba32f) uniform image2DArray uHistory;

layout(push_constant) uniform uuPushConstant {
	uint uIsAccumulate, uAccumulateCount, uAccumulationWindow, uHistoryWriteIndex;
};

void main() {
	vec3 color = subpassLoad(uColor).rgb;

	ivec2 coord = ivec2(gl_FragCoord.xy);
	if (uIsAccumulate != 0) {
		if (uAccumulationWindow != 0) {
			// Recompute from the actual last N samples. This is a finite window,
			// not an exponential average, and avoids long-running subtraction drift.
			// During warm-up only [0, count) contains initialized history texels.
			vec3 sum = color;
			for (uint layer = 0; layer < uAccumulateCount; ++layer) {
				if (layer != uHistoryWriteIndex)
					sum += imageLoad(uHistory, ivec3(coord, int(layer))).rgb;
			}
			imageStore(uHistory, ivec3(coord, int(uHistoryWriteIndex)), vec4(color, 0));
			color = sum / float(min(uAccumulateCount + 1, uAccumulationWindow));
		} else if (uAccumulateCount != 0) {
			color += imageLoad(uAccumulate, coord).rgb * float(uAccumulateCount);
			color /= float(uAccumulateCount + 1);
		}
	}
	// Publish the linear render at its native resolution for presentation and capture.
	imageStore(uAccumulate, coord, vec4(color, 0));
}
