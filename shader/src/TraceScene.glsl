#ifndef TRACE_SCENE_GLSL
#define TRACE_SCENE_GLSL

struct SceneIntersection {
	uint primitive_id, instance_id;
	vec2 barycentric;
};

bool IntersectScene(Ray ray, float limit, bool any_hit, out SceneIntersection hit) {
#ifdef NRC_EMULATOR
	// The reference APK loads one static OBJ instance. Use its actual index-buffer length.
	bool found = false;
	for (uint id = 0u; id < uint(uVertexIndices.length()) / 3u; ++id) {
		vec3 a = GetSceneVertex(0u, id, 0u);
		vec3 e1 = GetSceneVertex(0u, id, 1u) - a, e2 = GetSceneVertex(0u, id, 2u) - a;
		vec3 p = cross(ray.d, e2);
		float det = dot(e1, p);
		if (abs(det) < 1e-9) continue;
		vec3 delta = ray.o - a;
		float u = dot(delta, p) / det;
		if (u < 0.0 || u > 1.0) continue;
		vec3 q = cross(delta, e1);
		float v = dot(ray.d, q) / det, t = dot(e2, q) / det;
		if (v < 0.0 || u + v > 1.0 || t < T_MIN || t >= limit) continue;
		hit = SceneIntersection(id, 0u, vec2(u, v));
		if (any_hit) return true;
		limit = t;
		found = true;
	}
	return found;
#else
	rayQueryEXT query;
	uint flags = gl_RayFlagsOpaqueEXT | (any_hit ? gl_RayFlagsTerminateOnFirstHitEXT : 0u);
	rayQueryInitializeEXT(query, uTLAS, flags, 0xff, ray.o, T_MIN, ray.d, limit);
	while (rayQueryProceedEXT(query)) {}
	if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) return false;
	if (!any_hit) {
		hit.primitive_id = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true) + rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
		hit.instance_id = rayQueryGetIntersectionInstanceIdEXT(query, true);
		hit.barycentric = rayQueryGetIntersectionBarycentricsEXT(query, true);
	}
	return true;
#endif
}
#endif
