// chart index scheme:
// first coord:
// 0 - ambient space
// 1 - half-throat entry
// 2 - half-throat middle
// 3 - invalidated ray
// second coord:
// if first coord is 0, ambient space index
// if first coord is 1 or 2, half-throat index
// third coord:
// if first coord is 1 or 2, half-edge index

// As a general principle we probably wanna do transitions
// at the end of progress steps, bc we automatically get extra
// info about where the ray came from for free and we
// can maintain an invariant that all input rays are fully transitioned.

// We assume half-edges are stored in triangle triples

struct Camera {
    width: f32,
    height: f32,
    frame: mat3x3<f32>,
    frame_inv: mat3x3<f32>,
    centre: vec3<f32>,
    yfov: f32,
    chart_index: array<u32,3>,
}

struct TR3 {
  q: vec3f,
  v: vec3f,
}

const CHART_INDEX_SIZE = 3;
struct SituatedTR3 {
  chart_index: array<u32,CHART_INDEX_SIZE>,
  q: vec3f,
  v: vec3f,
}

fn situatedtr3_eq(a: SituatedTR3, b: SituatedTR3) -> bool {
  return a.chart_index[0] == b.chart_index[0] &&
    a.chart_index[1] == b.chart_index[1] &&
    a.chart_index[2] == b.chart_index[2] &&
    all(a.q == b.q) &&
    all(a.v == b.v);
}

struct HalfEdge {
    index: u32,
    prev: u32,
    next: u32,
    twin: u32,
    vertex: vec3f,
}

struct Mesh {
    he_lo_index: u32,
    he_hi_index: u32,
}

struct HalfThroat {
    ltg: mat4x4f,
    gtl: mat4x4f,
    index: u32,
    ambient_index: u32,
    twin_index: u32,
    mesh: Mesh,
    mid_t: f32, // other side must have same mid_t
    hi_t: f32, // 2*mid_t - 1 > hi_t > mid_t
    throat_to_mid_t: f32, // mid_t > throat_to_mid_t > mid_to_throat_t > 1
    mid_to_throat_t: f32,
}

struct LocalTriangle {
    he: HalfEdge,
    p1: vec3f,
    p2: vec3f,
}

struct TriangleTrafoState {
    lt: LocalTriangle,
    pos: vec3f,
    delta: vec3f,
    trafo: mat3x3f // old coords to new coords cumulative
}

struct TriangleIntersect {
    tuv: vec3f,
    he: HalfEdge,
    ht_index: u32,
}

fn local_triangle_from_halfedge(he: HalfEdge) -> LocalTriangle {
    let n = half_edges[he.next];
    let p = half_edges[he.prev];
    let dr = n.vertex - he.vertex;
    let dl = p.vertex - he.vertex;
    let base = length(dr);
    let xn = normalize(dr);
    let tip_x = dot(dl, xn);
    let tip_y = sqrt(dot(dl, dl) - tip_x * tip_x);
    return LocalTriangle(he, vec3f(base,0,1), vec3f(tip_x, tip_y, 1));
}

fn small_step(tts: TriangleTrafoState) -> TriangleTrafoState {
    let t0 = vec3f(-tts.lt.p1.xy, 0); // twin 0, based at p1
    let t1 = tts.lt.p1 - tts.lt.p2; // twin 1, based at p2
    let t2 = vec3f(tts.lt.p2.xy, 0); // twin 2, based at 0
    let extrapolated_new_pos = tts.pos + tts.delta;
    let h0 = cross(t0, extrapolated_new_pos - tts.lt.p1).z;
    let h1 = cross(t1, extrapolated_new_pos - tts.lt.p2).z;
    let h2 = cross(t2, extrapolated_new_pos - vec3f(0,0,1)).z;
    if (h0 > 0 || h1 > 0 || h2 > 0) { // we step outside the triangle
        let d0 = cross(vec3f(0,0,1) - tts.pos, tts.delta).z;
        let d1 = cross(tts.lt.p1 - tts.pos, tts.delta).z;
        let d2 = cross(tts.lt.p2 - tts.pos, tts.delta).z;
        var edge_vec = vec3f();
        var new_he = HalfEdge();
        if (d0 >= 0 && d1 < 0) {
            // crosses 0
            edge_vec = t0;
            new_he = half_edges[tts.lt.he.twin];
        } else if (d1 >= 0 && d2 < 0) {
            // crosses 1
            edge_vec = t1;
            new_he = half_edges[half_edges[tts.lt.he.next].twin];
        } else {
            // crosses 2
            edge_vec = t2;
            new_he = half_edges[half_edges[tts.lt.he.prev].twin];
        }
        let pos_vec = vec3f(edge_vec.xy, 1);
        let l0le = length(t0) * length(edge_vec);
        let cos_angle = -dot(t0, edge_vec)/l0le;
        let sin_angle = -cross(t0,edge_vec).z/l0le;
        let lin_transform = mat3x3f(vec3f(cos_angle, -sin_angle, 0), vec3f(sin_angle, cos_angle, 0), vec3f(0,0,1));
        let lcol = -lin_transform * pos_vec;
        let transform = mat3x3f(lin_transform[0], lin_transform[1], lcol);
        let new_triangle = local_triangle_from_halfedge(new_he);
        let modified_delta = transform * tts.delta;
        let modified_pos = transform * tts.pos;
        // y(modified_pos + t * modified_delta) = 0
        // t = -y(modified_pos)/y(modified_delta)
        let t = -modified_pos.y/modified_delta.y;
        let new_pos = vec3f(clamp((modified_pos + t*modified_delta).x, 0.0, new_triangle.p1.x), 0, 1);
        let new_delta = (1-t)*modified_delta;
        return TriangleTrafoState(new_triangle, new_pos, new_delta, transform * tts.trafo);
    }
    return TriangleTrafoState(tts.lt, tts.pos + tts.delta, vec3f(), tts.trafo);
}

const STEP_BOUND: u32 = 50;
fn big_step(tts: TriangleTrafoState) -> TriangleTrafoState {
    var w = tts;
    // for (var k: u32 = 0; k < STEP_BOUND; k++) {
      // if (any(abs(w.delta) > vec3f())) {
        // break;
      // }
      w = small_step(w);
    // }
    return w;
}

fn fragpos_to_ray(camera: Camera, pos: vec2f)->SituatedTR3 {
    let ray_coords = normalize(vec3f( // Note the normalization - if camera frame is orthonormal, ray will be also
        (pos.x / camera.width - 0.5) * (camera.width / camera.height),
        pos.y / camera.height - 0.5,
        0.5 / (tan(camera.yfov / 2.0))
    ));
    let ray = camera.frame * ray_coords;
    return SituatedTR3(camera.chart_index, camera.centre, ray);
}

const simple_throat_points = array(
    vec2f(1.0, 0.0), // left is scale param, right is "height"
    vec2f(0.5, 0.0),
    vec2f(0.5, 0.0),
    vec2f(0.5, 0.5)
);

fn simple_cubic(t: f32) -> vec2f {
    return cubic(simple_throat_points, t);
}

fn simple_cubic_velocity(t: f32) -> vec2f {
    return cubic_velocity(simple_throat_points, t);
}

fn simple_cubic_accel(t: f32) -> vec2f {
    return cubic_accel(simple_throat_points, t);
}

fn extended_cubic(t: f32) -> vec2f {
    if (t < 0) {
        return t * simple_cubic_velocity(0.0) + simple_cubic(0.0);
    } else if (t > 1) {
        return (t - 1) * simple_cubic_velocity(1.0) + simple_cubic(1.0);
    } else {
        return simple_cubic(t);
    }
}

fn extended_cubic_velocity(t: f32) -> vec2f {
    return simple_cubic_velocity(clamp(t, 0.0, 1.0));
}

fn extended_cubic_accel(t: f32) -> vec2f {
    if (t < 0 || t > 1) {
        return vec2f();
    } else {
        return simple_cubic_accel(t);
    }
}

fn cubic(points: array<vec2f, 4>, t: f32) -> vec2f {
    var a = mix(points[0], points[1], t);
    var b = mix(points[1], points[2], t);
    var c = mix(points[2], points[3], t);
    a = mix(a,b,t);
    b = mix(b,c,t);
    a = mix(a,b,t);
    return a;
}

fn cubic_velocity(points: array<vec2f, 4>, t: f32) -> vec2f {
    var a = points[1]-points[0];
    var b = points[2]-points[1];
    var c = points[3]-points[2];
    a = mix(a,b,t);
    b = mix(b,c,t);
    a = mix(a,b,t);
    return 3*a;
}

fn cubic_accel(points: array<vec2f, 4>, t: f32) -> vec2f {
    var a = points[2] - 2 * points[1] + points[0];
    var b = points[3] - 2 * points[2] + points[1];
    a = mix(a,b,t);
    return 6*a;
}

fn half_edge_to_embedded_basis(he: HalfEdge) -> mat4x4f {
    let p0 = he.vertex;
    let p1 = half_edges[he.next].vertex;
    let p2 = half_edges[he.prev].vertex;
    let b1 = normalize(p1-p0);
    let e2 = p2-p0;
    let b2 = normalize(e2 - dot(e2,b1)*b1);
    let b3 = cross(b1,b2);
    return mat4x4f(vec4f(b1,0), vec4f(b2,0), vec4f(b3,0), vec4f(p0,1));
}

fn invert_orthobasis(ob: mat4x4f) -> mat4x4f {
    let corner = mat3x3f(ob[0].xyz, ob[1].xyz, ob[2].xyz);
    let tcorner = transpose(corner);
    let antitranslation = -tcorner * ob[3].xyz;
    return mat4x4f(vec4f(tcorner[0],0), vec4f(tcorner[1],0), vec4f(tcorner[2],0), vec4f(antitranslation, 1));
}

fn jac_tilde(he: HalfEdge, t: f32) -> mat3x4f {
    let eb = half_edge_to_embedded_basis(he);
    let vel = extended_cubic_velocity(t);
    let p = dot(eb[2].xyz, eb[3].xyz) * eb[2];
    let c2 = vel.x * p + vec4f(0,0,0,vel.y);
    return mat3x4f(eb[0], eb[1], c2);
}

fn jac_tilde_pinv(he: HalfEdge, t: f32) -> mat4x3f {
    var jt = jac_tilde(he, t);
    jt[2] = jt[2] / dot(jt[2], jt[2]);
    return transpose(jt);
}

fn uv_to_mesh_pos(he: HalfEdge, uv: vec2f) -> vec4f {
    let eb = half_edge_to_embedded_basis(he);
    return eb[3] + uv.x * eb[0] + uv.y * eb[1];
}

fn uvt_to_4pos(he: HalfEdge, uvt: vec3f) -> vec4f {
    let eb = half_edge_to_embedded_basis(he);
    let sc = extended_cubic(uvt.z);
    let mesh_pos = eb[3] + uvt.x * eb[0] + uvt.y * eb[1];
    return vec4f(sc.x * mesh_pos.xyz, sc.y);
}

fn ortho_to_uvt(he: HalfEdge, uvt: vec3f) -> mat3x3f { // JL = Jtilde, this is L
    let eb = half_edge_to_embedded_basis(he);
    let sc = extended_cubic(uvt.z);
    let vel = extended_cubic_velocity(uvt.z);
    let inv_scale = 1/sc.x;
    let a = dot(eb[3], eb[0]) + uvt.x;
    let b = dot(eb[3], eb[1]) + uvt.y;
    return mat3x3f(
        inv_scale, 0, 0,
        0, inv_scale, 0,
        -a * vel.x * inv_scale, -b * vel.x * inv_scale, 1
    );
}

fn jac(he: HalfEdge, uvt: vec3f) -> mat3x4f {
    let eb = half_edge_to_embedded_basis(he);
    let sc = extended_cubic(uvt.z);
    let vel = extended_cubic_velocity(uvt.z);
    let mesh_pos = eb[3] + uvt.x * eb[0] + uvt.y * eb[1];
    return mat3x4f(sc.x * eb[0], sc.x * eb[1], vec4f(vel.x * mesh_pos.xyz, vel.y));
}

fn jac_pinv(he: HalfEdge, uvt: vec3f) -> mat4x3f {
    return ortho_to_uvt(he, uvt) * jac_tilde_pinv(he, uvt.z);
}

fn djac_dt(he: HalfEdge, uvt: vec3f, uvt_vel: vec3f) -> mat3x4f {
    let vel = extended_cubic_velocity(uvt.z);
    let acc = extended_cubic_accel(uvt.z);
    let eb = half_edge_to_embedded_basis(he);
    let mesh_pos = (eb[3] + uvt.x * eb[0] + uvt.y * eb[1]).xyz;
    return mat3x4f(
        vel.x * uvt_vel.z * eb[0],
        vel.x * uvt_vel.z * eb[1],
        vec4f(
            acc.x * uvt_vel.z * mesh_pos
                + vel.x * uvt_vel.x * eb[0].xyz
                + vel.x * uvt_vel.y * eb[1].xyz,
            acc.y * uvt_vel.z
        )
    );
}

// Justifying the move to local coords as coherent with global coords is a little tricky.
// Seems to depend on the fact that our transform is the same affine map everywhere.
// The metric is not necessarily preserved but the condition for parallel transport is,
// I think, so the connection is preserved altogether.
// So we end up tracing in a different metric but the lines stay the same.
fn half_throat_entry(global_ray: SituatedTR3, is: TriangleIntersect) -> SituatedTR3 {
    let ht = half_throats[is.ht_index];
    let throat_local_ray_4vel = (ht.gtl * vec4f(global_ray.v, 0));
    let embedded_basis = half_edge_to_embedded_basis(is.he);
    let e1 = half_edges[is.he.next].vertex - is.he.vertex;
    let e2 = half_edges[is.he.prev].vertex - is.he.vertex;
    let throat_local_tri_centered_intersection_pos = is.tuv.y * e1 + is.tuv.z * e2;
    let chart_local_pos = vec3f(
        dot(throat_local_tri_centered_intersection_pos, embedded_basis[0].xyz),
        dot(throat_local_tri_centered_intersection_pos, embedded_basis[1].xyz),
        0.0,
    ); // cubic param last
    let chart_local_ray_vel = jac_pinv(is.he, chart_local_pos) * throat_local_ray_4vel;
    return SituatedTR3(
        array(1, ht.index, is.he.index),
        chart_local_pos,
        chart_local_ray_vel,
    );
}

// For good hysteresis, use this when the spline param is a little negative
fn half_throat_exit(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    let he = half_edges[throat_patch_ray.chart_index[2]];
    let throat_local_4pos = uvt_to_4pos(he, throat_patch_ray.q);
    let throat_local_4vel = jac(he, throat_patch_ray.q) * throat_patch_ray.v;
    let throat_local_pos = vec4f(throat_local_4pos.xyz, 1); // we assume throat_local_4pos.w is 0, i.e. we're in the flat region
    let throat_local_vel = vec4f(throat_local_4vel.xyz, 0); // same as above
    let ht = half_throats[throat_patch_ray.chart_index[1]];
    let ltg = ht.ltg;
    let global_pos = ltg * throat_local_pos;
    let global_vel = ltg * throat_local_vel;
    return SituatedTR3(array(0, ht.ambient_index, 0), global_pos.xyz, global_vel.xyz);
}

fn mid_throat_entry(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    return SituatedTR3(array(2, throat_patch_ray.chart_index[1], throat_patch_ray.chart_index[2]), throat_patch_ray.q, throat_patch_ray.v);
}

fn mid_throat_exit(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    return SituatedTR3(array(1, throat_patch_ray.chart_index[1], throat_patch_ray.chart_index[2]), throat_patch_ray.q, throat_patch_ray.v);
}

fn mid_throat_transition(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    let ht = half_throats[throat_patch_ray.chart_index[1]];
    return SituatedTR3(
        array(throat_patch_ray.chart_index[0], ht.twin_index, throat_patch_ray.chart_index[2]),
        vec3f(throat_patch_ray.q.xy, 2*ht.mid_t - throat_patch_ray.q.z),
        vec3f(throat_patch_ray.v.xy, -throat_patch_ray.v.z)
    );
}

fn phase_vel(he: HalfEdge, qv: mat2x3f) -> mat2x3f {
    let djdt = djac_dt(he, qv[0], qv[1]);
    let jpinv = jac_pinv(he, qv[0]);
    return mat2x3f(qv[1], -jpinv * (djdt * qv[1]));
}

// only rk4 + triangle steppin
fn throat_step(throat_patch_ray: SituatedTR3, dt: f32) -> SituatedTR3 {
    let he = half_edges[throat_patch_ray.chart_index[2]];
    let qv0 = mat2x3f(throat_patch_ray.q, throat_patch_ray.v);
    let k1 = phase_vel(he, qv0);
    // let k2 = phase_vel(he, qv0 + (dt/2)*k1);
    // let k3 = phase_vel(he, qv0 + (dt/2)*k2);
    // let k4 = phase_vel(he, qv0 + (dt)*k3);
    // let delta = (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    let delta = dt * k1;
    let mesh_pos_0_affine = vec3f(qv0[0].xy, 1);
    let mesh_delta_affine = vec3f(delta[0].xy, 0);
    let tts = TriangleTrafoState(local_triangle_from_halfedge(he), mesh_pos_0_affine, mesh_delta_affine, mat3x3f(1,0,0,0,1,0,0,0,1));
    let ntts = big_step(tts);
    // if bounded, need to invalidate here
    let qv1_old_coords = qv0 + delta;
    let mesh_qv_1_affine_old = mat2x3f(vec3f(qv1_old_coords[0].xy, 1), vec3f(qv1_old_coords[1].xy, 0));
    let mesh_qv_1_affine_new = ntts.trafo * mesh_qv_1_affine_old;
    let qv1_new_coords = mat2x3f(
        vec3f(mesh_qv_1_affine_new[0].xy, qv1_old_coords[0].z),
        vec3f(mesh_qv_1_affine_new[1].xy, qv1_old_coords[1].z)
    ); // ideally, qv1_new_coords[0].xy == ntts.pos.xy, but numerically idk if it works out
    // caveat emptor
    let prelim = SituatedTR3(throat_patch_ray.chart_index, qv1_new_coords[0], qv1_new_coords[1]);
    let ht = half_throats[prelim.chart_index[1]];
    if (prelim.q.z > ht.throat_to_mid_t) {
        return mid_throat_entry(prelim);
    } else if (prelim.q.z < 0) { // tune this constant maybe
        return half_throat_exit(prelim);
    }
    return prelim;
}

// May invalidate
fn midthroat_traverse(throat_patch_ray: SituatedTR3, t_length_bound: f32) -> SituatedTR3 {
    let ht = half_throats[throat_patch_ray.chart_index[1]];
    let hi_target_t = 2*ht.mid_t - 1.0;
    let lo_target_t = 1.0;
    if (throat_patch_ray.v.z == 0) {
        return SituatedTR3(array(3,throat_patch_ray.chart_index[1], throat_patch_ray.chart_index[2]), throat_patch_ray.q, throat_patch_ray.v);
    }
    let target_t = select(lo_target_t, hi_target_t, throat_patch_ray.v.z > 0);
    let l = (target_t - throat_patch_ray.q.z)/throat_patch_ray.v.z;
    if (abs(l) > t_length_bound) { // maybe should be abs(l * throat_patch_ray.v.z)
        return SituatedTR3(array(3,throat_patch_ray.chart_index[1], throat_patch_ray.chart_index[2]), throat_patch_ray.q, throat_patch_ray.v);
    }
    let delta = l * throat_patch_ray.v;
    let mesh_qv_0_affine = mat2x3f(vec3f(throat_patch_ray.q.xy, 1), vec3f(throat_patch_ray.v.xy, 0));
    let he = half_edges[throat_patch_ray.chart_index[2]];
    let tts = TriangleTrafoState(local_triangle_from_halfedge(he), mesh_qv_0_affine[0], vec3f(delta.xy, 0), mat3x3f(1,0,0,0,1,0,0,0,1));
    let ntts = big_step(tts);
    // if bounded, need to invalidate here
    let mesh_qv_1_affine = ntts.trafo * mesh_qv_0_affine;
    let new_q = vec3f(mesh_qv_1_affine[0].xy, throat_patch_ray.q.z + delta.z);
    let new_v = vec3f(mesh_qv_1_affine[1].xy, throat_patch_ray.v.z);
    let prelim_tr3 = SituatedTR3(throat_patch_ray.chart_index, new_q, new_v);
    if (throat_patch_ray.v.z > 0) {
        return mid_throat_exit(mid_throat_transition(prelim_tr3));
    }
    return mid_throat_exit(prelim_tr3);
}

fn push_ray_step(ray: SituatedTR3) -> SituatedTR3 {
    let dt = 0.01;
    let t_length_bound = 50.0;
    if (ray.chart_index[0] == 0) {
        return process_ambient_ray(ray);
    } else if (ray.chart_index[0] == 1) {
        // return throat_step(ray, dt);
        return ray;
    } else if (ray.chart_index[0] == 2) {
        // return midthroat_traverse(ray, t_length_bound);
        return ray;
    } else {
        return ray;
    }
}

fn push_ray(ray: SituatedTR3, max_iter: u32) -> SituatedTR3 {
    var cur = ray;
    for (var k: u32 = 0; k < max_iter; k++) {
        let newRay = push_ray_step(cur);
        if (situatedtr3_eq(newRay, cur)) {
            return cur;
        }
        cur = newRay;
    }
    return cur;
}

// Helper function to perform ray-triangle intersection
// Claude-gen
fn ray_triangle_intersect(ray_origin: vec3f, ray_dir: vec3f, he: HalfEdge) -> vec3f {
    let e1 = half_edges[he.next].vertex - he.vertex;
    let e2 = half_edges[he.prev].vertex - he.vertex;
    
    let h = cross(ray_dir, e2);
    let a = dot(e1, h);
    
    // Ray is parallel to triangle
    if (abs(a) < 1e-8) {
        return vec3f(-1.0, 0.0, 0.0); // Invalid intersection
    }
    
    let f = 1.0 / a;
    let s = ray_origin - he.vertex;
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return vec3f(-1.0, 0.0, 0.0); // Invalid intersection
    }
    
    let q = cross(s, e1);
    let v = f * dot(ray_dir, q);
    
    if (v < 0.0 || u + v > 1.0) {
        return vec3f(-1.0, 0.0, 0.0); // Invalid intersection
    }
    
    let t = f * dot(e2, q);
    
    if (t > 1e-8) { // Ray intersection
        return vec3f(t, u, v);
    } else {
        return vec3f(-1.0, 0.0, 0.0); // Invalid intersection
    }
}

// Function to find intersection with a specific half-throat mesh
// Claude-gen
fn intersect_half_throat_mesh(ray: SituatedTR3, ht: HalfThroat) -> TriangleIntersect {
    var closest_intersection: TriangleIntersect;
    var closest_t = 1e30;
    var found_intersection = false;
    
    // Transform ray to throat local coordinates
    let throat_local_origin = (ht.gtl * vec4f(ray.q, 1.0)).xyz;
    let throat_local_dir = (ht.gtl * vec4f(ray.v, 0.0)).xyz;
    
    // Iterate through all triangles in the mesh
    for (var he_idx = ht.mesh.he_lo_index; he_idx < ht.mesh.he_hi_index; he_idx += 3u) {
        let he = half_edges[he_idx];
        
        // Test intersection with this triangle
        let tuv = ray_triangle_intersect(throat_local_origin, throat_local_dir, he);
        
        if (tuv.x > 0.0 && tuv.x < closest_t) {
            closest_t = tuv.x;
            closest_intersection = TriangleIntersect(tuv, he, ht.index);
            found_intersection = true;
        }
    }
    
    if (!found_intersection) {
        // Return invalid intersection
        closest_intersection.tuv = vec3f(-1.0, 0.0, 0.0);
        closest_intersection.he = HalfEdge(0u, 0u, 0u, 0u, vec3f(0.0));
        closest_intersection.ht_index = 0u;
    }
    
    return closest_intersection;
}

// Function to find the closest intersection among all half-throats in the ambient space
// Claude-gen
fn find_closest_half_throat_intersection(ray: SituatedTR3) -> TriangleIntersect {
    var closest_intersection: TriangleIntersect;
    var closest_t = 1e30;
    var found_intersection = false;
    
    let ambient_index = ray.chart_index[1];
    
    // Determine the range of half-throats for this ambient space
    let start_idx = select(0u, half_throat_range_end_index[ambient_index - 1u], ambient_index > 0u);
    let end_idx = half_throat_range_end_index[ambient_index];
    
    // Iterate through half-throats belonging to this ambient space
    for (var ht_idx = start_idx; ht_idx < end_idx; ht_idx++) {
        let ht = half_throats[ht_idx];
        let intersection = intersect_half_throat_mesh(ray, ht);
        
        if (intersection.tuv.x > 0.0 && intersection.tuv.x < closest_t) {
            closest_t = intersection.tuv.x;
            closest_intersection = intersection;
            found_intersection = true;
        }
    }
    
    if (!found_intersection) {
        // Return invalid intersection
        closest_intersection.tuv = vec3f(-1.0, 0.0, 0.0);
        closest_intersection.he = HalfEdge(0u, 0u, 0u, 0u, vec3f(0.0));
        closest_intersection.ht_index = 0u;
    }
    
    return closest_intersection;
}

// Main function for ambient ray processing
// Claude-gen
fn process_ambient_ray(ray: SituatedTR3) -> SituatedTR3 {
    // Only process rays that are in ambient space (chart_index[0] == 0)
    if (ray.chart_index[0] != 0u) {
        return ray; // Return unchanged if not in ambient space
    }
    
    // Find the closest intersection with any half-throat mesh
    let intersection = find_closest_half_throat_intersection(ray);
    
    // If no intersection found, return the original ray
    if (intersection.tuv.x < 0.0) {
        return ray;
    }
    
    // Use the existing function to transform to half-throat entry coordinates
    return half_throat_entry(ray, intersection);
}


// Two general situations:
// 1. Freestanding ray
// 2. Ray with intersection info
// We would like to treat the latter as a transient state, so we instantly eat it
// Processing principle:
// 1. Given a freestanding ray, perform any applicable transitions
// 2. Advance the ray
// 3. Terminate when invalid, no progress, or run out of slack
// Given this principle, advancing an ambient ray must also perform its intersection transition on its own

@group(0) @binding(0) var<storage> camera : Camera;
@group(0) @binding(1) var skybox_sampler: sampler;
@group(1) @binding(0) var skybox_textures: texture_cube_array<f32>;
@group(2) @binding(0) var<storage> half_throats: array<HalfThroat>;
@group(2) @binding(1) var<storage> half_edges: array<HalfEdge>;
@group(2) @binding(2) var<storage> half_throat_range_end_index: array<u32>;
@group(3) @binding(0) var<uniform> screen_size: vec2<u32>;
@group(3) @binding(1) var<storage, read_write> screen_data: array<vec4f>;

@vertex
fn vtx_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4f {
  var pos: vec2f;
  switch vertex_index {
    case 0u: { pos = vec2(-1.0, -1.0); }
    case 1u: { pos = vec2(3.0, -1.0); }
    case 2u: { pos = vec2(-1.0, 3.0); }
    default: { pos = vec2(0.0, 0.0); }
  }
  return vec4f(pos, 0, 1);
}

@fragment
fn frag_main(@builtin(position) in : vec4<f32>) -> @location(0) vec4f {
    let pos = vec2<u32>(in.xy);
    let lin_color = screen_data[pos.y * screen_size.x + pos.x];
    return lin_color;
}

@compute @workgroup_size(16, 16)
fn compute_main(@builtin(global_invocation_id) gid: vec3<u32>)
{
  let cid: vec2<u32> = clamp(gid.xy, vec2<u32>(0,0), screen_size - vec2<u32>(1,1));
  let frag_pos = vec2<f32>(cid) + vec2<f32>(0.5, 0.5);
  let ray = fragpos_to_ray(camera, frag_pos);
  let pushed = push_ray_step(ray);

  var skybox_color: vec4f = vec4f(0,0,0,1);

  switch pushed.chart_index[0] {
    case 0u: {
      // Ambient space - use ambient space index as array layer
      let ambient_index = pushed.chart_index[1];
      // skybox_color = textureSample(skybox_textures, skybox_sampler, pushed.v, ambient_index);
      switch ambient_index {
        case 0u: {
          skybox_color = vec4f(1,0,0,1);
        }
        default: {
          skybox_color = vec4f(0,0,1,1);
        }
      }
    }
    default: {
      // invalid case
      skybox_color = vec4f(1.0, 1.0, 0.0, 1.0); // Yellow fallback
    }
  }
  screen_data[cid.y * screen_size.x + cid.x] = skybox_color;
}