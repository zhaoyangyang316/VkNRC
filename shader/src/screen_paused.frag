#version 450

layout(binding = 0, rgba32f) readonly uniform image2D uFrozen;
layout(push_constant) uniform uuPushConstant { uvec2 uExtent; };
layout(location = 0) out vec4 oScreen;

vec3 ToneMapFilmic_Hejl2015(in const vec3 hdr, in const float white_pt) {
	vec4 vh = vec4(hdr, white_pt);
	vec4 va = (1.425 * vh) + 0.05;
	vec4 vf = (vh * va + 0.004) / ((vh * (va + 0.55) + 0.0491)) - 0.0821;
	return vf.rgb / vf.w;
}

void main() {
	// Fit the retained frame to a resized viewport without changing its aspect
	// ratio or accessing texels outside the original image.
	ivec2 source_extent = imageSize(uFrozen);
	float scale = min(float(uExtent.x) / source_extent.x, float(uExtent.y) / source_extent.y);
	vec2 offset = (vec2(uExtent) - vec2(source_extent) * scale) * 0.5;
	ivec2 coord = ivec2(floor((gl_FragCoord.xy - offset) / scale));
	if (any(lessThan(coord, ivec2(0))) || any(greaterThanEqual(coord, source_extent))) {
		oScreen = vec4(0, 0, 0, 1);
		return;
	}
	vec3 color = imageLoad(uFrozen, coord).rgb;
	vec3 screen = pow(max(ToneMapFilmic_Hejl2015(color, 3.2), vec3(0)), vec3(1 / 2.2));
	oScreen = vec4(screen, 1.0);
}
