// chart index scheme:
// first coord:
// 0 - ambient space
// 1 - half-throat entry
// 2 - half-throat middle
// second coord:
// if first coord is 0, ambient space index
// if first coord is 1 or 2, half-throat index
// third coord:
// if first coord is 1 or 2, half-edge index

struct Camera {
    width: f32,
    height: f32,
    frame: mat3x3<f32>,
    frame_inv: mat3x3<f32>,
    centre: vec3<f32>,
    yfov: f32,
    chart_index: vec3<u32>,
}

struct TR3 {
  q: vec3f,
  v: vec3f,
}

struct SituatedTR3 {
  chart_index: vec3<u32>,
  q: vec3f,
  v: vec3f,
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
    trafo: mat3x3f
}

struct TriangleIntersect {
    tuv: vec3f,
    he: HalfEdge,
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

fn big_step(tts: TriangleTrafoState) -> TriangleTrafoState {
    var w = tts;
    while (any(abs(w.delta) > vec3f())) {
        w = small_step(w);
    }
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

// Justifying the move to local coords as coherent with global coords is a little tricky.
// Seems to depend on the fact that our transform is the same affine map everywhere.
// The metric is not necessarily preserved but the condition for parallel transport is,
// I think, so the connection is preserved altogether.
// So we end up tracing in a different metric but the lines stay the same.
fn half_throat_entry(ht: HalfThroat, global_ray: SituatedTR3, is: TriangleIntersect) -> SituatedTR3 {
    let throat_local_ray_vel_affine = (ht.gtl * vec4f(global_ray.v, 0));
    let embedded_basis = half_edge_to_embedded_basis(is.he);
    let e1 = half_edges[is.he.next].vertex - is.he.vertex;
    let e2 = half_edges[is.he.prev].vertex - is.he.vertex;
    let sv0 = simple_cubic_velocity(0);
    let tri_local_ray_vel = vec3f(
        dot(throat_local_ray_vel_affine, embedded_basis[0]),
        dot(throat_local_ray_vel_affine, embedded_basis[1]),
        dot(throat_local_ray_vel_affine, embedded_basis[2])
    );
    let chart_local_ray_vel = vec3f(tri_local_ray_vel.xy, -tri_local_ray_vel.z/length(sv0)); // cubic param last
    let throat_local_tri_centered_intersection_pos = is.tuv.y * e1 + is.tuv.z * e2;
    let chart_local_pos = vec3f(
        dot(throat_local_tri_centered_intersection_pos, embedded_basis[0].xyz),
        dot(throat_local_tri_centered_intersection_pos, embedded_basis[1].xyz),
        0.0,
    ); // cubic param last
    return SituatedTR3(
        vec3<u32>(1, ht.index, is.he.index),
        chart_local_pos,
        chart_local_ray_vel,
    );
}

// TODO: redo this with hysteresis
fn half_throat_exit(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    // spline param for pos is zero
    let he = half_edges[throat_patch_ray.chart_index[2]];
    let embedded_basis = half_edge_to_embedded_basis(he);
    let sv0 = simple_cubic_velocity(0);
    let throat_local_pos = embedded_basis * vec4f(throat_patch_ray.q, 1); // only works if spline param is zero
    let vel_in_embedded_basis = vec4f(throat_patch_ray.v.xy, -length(sv0) * throat_patch_ray.v.z, 0);
    let throat_local_vel = embedded_basis * vel_in_embedded_basis;
    let ht = half_throats[throat_patch_ray.chart_index[1]];
    let ltg = ht.ltg;
    let global_pos = ltg * throat_local_pos;
    let global_vel = ltg * throat_local_vel;
    return SituatedTR3(vec3<u32>(0, ht.ambient_index, 0), global_pos.xyz, global_vel.xyz);
}

fn mid_throat_entry(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    return SituatedTR3(vec3<u32>(2, throat_patch_ray.chart_index.yz), throat_patch_ray.q, throat_patch_ray.v);
}

fn mid_throat_exit(throat_patch_ray: SituatedTR3) -> SituatedTR3 {
    return SituatedTR3(vec3<u32>(1, throat_patch_ray.chart_index.yz), throat_patch_ray.q, throat_patch_ray.v);
}

@binding(0) @group(0) var<uniform> camera : Camera;
@binding(0) @group(1) var<storage> half_throats: array<HalfThroat>;
@binding(0) @group(2) var<storage> half_edges: array<HalfEdge>;

@vertex
fn vtx_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4f {
  const pos = array(
    vec2( -1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2( -1.0, 3.0)
  );

  return vec4f(pos[vertex_index], 0, 1);
}

@fragment
fn frag_main(@builtin(position) in : vec4<f32>) -> @location(0) vec4f {
  let ray = fragpos_to_ray(camera, in.xy);
  return vec4f(1.0, 1.0, 0.0, 1.0);
}