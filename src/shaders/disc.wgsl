struct Camera {
    width: f32,
    height: f32,
    frame: mat3x3<f32>,
    frame_inv: mat3x3<f32>,
    centre: vec3<f32>,
    yfov: f32,
    chart_index: u32,
}

struct TR3 {
  q: vec3f,
  v: vec3f,
}

struct SituatedTR3 {
  chart_index: u32,
  q: vec3f,
  v: vec3f,
}

struct HalfEdge {
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
    base: f32,
    tip: vec2f,
}

struct TrianglePosState {
    he: HalfEdge,
    lt: LocalTriangle,
    pos: vec2f,
}

struct QuadIntersectRes {
    interps: vec2f,
    intersects: bool,
}

// (1-u)a + ub = (1-v)c + vd
// then interps.xy = (u,v)
fn quad_intersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> QuadIntersectRes {
    let ba = b-a;
    let cd = c-d;
    let ca = c-a;
    let dd = ba.x * cd.y - ba.y * cd.x;
    if (dd == 0) {
        return QuadIntersectRes(vec2f(), false);
    }
    let mm = mat2x2f(cd.y, -ba.y, -cd.x, ba.x) * (1/dd);
    return QuadIntersectRes(mm * ca, true);
}

fn intersection_is_segmental(ir: QuadIntersectRes) -> bool {
    return 0 <= ir.interps.x && 0 <= ir.interps.y && ir.interps.x <= 1 && ir.interps.y <= 1;
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
    let tip = vec2f(tip_x, tip_y);
    return LocalTriangle(base, tip);
}

fn local_triangle_step(tps: TrianglePosState, delta: vec2f) -> TrianglePosState {
    var more_to_traverse = true;
    var itps = tps;
    var idelta = delta;
    while (more_to_traverse) {
        let rc = vec2f(itps.lt.base, 0.0);
        let newpos = itps.pos + idelta;
        let i0 = quad_intersect(itps.pos, newpos, vec2f(), rc);
        if (i0.intersects && intersection_is_segmental(i0)) {
            let new_he = half_edges[itps.he.twin];
            let new_local_triangle = local_triangle_from_halfedge(new_he);
            let new_pos = vec2f((1-i0.interps.y) * new_local_triangle.base, 0);
            let new_idelta = -(1-i0.interps.x)*idelta;
            itps = TrianglePosState(new_he, new_local_triangle, new_pos);
            idelta = new_idelta;
            continue;
        }
        let i1 = quad_intersect(itps.pos, newpos, rc, itps.lt.tip);
        if (i1.intersects && intersection_is_segmental(i1)) {
            let base_he = half_edges[itps.he.next];
            let new_he = half_edges[base_he.twin];
            let new_local_triangle = local_triangle_from_halfedge(new_he);
            let new_pos = vec2f((1-i0.interps.y) * new_local_triangle.base, 0);
            let rot_angle_cos = dot(rc, rc-itps.lt.tip)/(itps.lt.base * length(rc-itps.lt.tip));
            let rot_angle_sin = sqrt(1-rot_angle_cos*rot_angle_cos);
            let rotmat = mat2x2f(rot_angle_cos, rot_angle_sin, -rot_angle_sin, rot_angle_cos);
            let new_idelta = (1-i0.interps.x) * (rotmat * idelta);
            itps = TrianglePosState(new_he, new_local_triangle, new_pos);
            idelta = new_idelta;
            continue;
        }
        let i2 = quad_intersect(itps.pos, newpos, itps.lt.tip, vec2f());
        if (i2.intersects && 0 <= i2.interps.x && i2.interps.x <= 1) {
            let base_he = half_edges[itps.he.prev];
            let new_he = half_edges[base_he.twin];
            let new_local_triangle = local_triangle_from_halfedge(new_he);
            let new_pos = vec2f((1-i0.interps.y) * new_local_triangle.base, 0);
            let rot_angle_cos = dot(rc, itps.lt.tip)/(itps.lt.base * length(itps.lt.tip));
            let rot_angle_sin = -sqrt(1-rot_angle_cos*rot_angle_cos);
            let rotmat = mat2x2f(rot_angle_cos, rot_angle_sin, -rot_angle_sin, rot_angle_cos);
            let new_idelta = (1-i0.interps.x) * (rotmat * idelta);
            itps = TrianglePosState(new_he, new_local_triangle, new_pos);
            idelta = new_idelta;
            continue;
        }
        more_to_traverse = false;
    }
    return TrianglePosState(itps.he, itps.lt, itps.pos + idelta);
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

struct HalfThroatTV {
    half_throat: HalfThroat,
    tps: TrianglePosState,
    t: f32,
    vel: vec3f,
}

// Justifying the move to local coords as coherent with global coords is a little tricky.
// Seems to depend on the fact that our transform is the same affine map everywhere.
// The metric is not necessarily preserved but the condition for parallel transport is,
// I think, so the connection is preserved altogether.
// So we end up tracing in a different metric but the lines stay the same.
fn half_throat_entry(ht: HalfThroat, global_ray: vec3f, is: TriangleIntersect) -> HalfThroatTV {
    let local_ray = (ht.gtl * vec4f(global_ray, 0)).xyz;
    let e1 = half_edges[is.he.next].vertex - is.he.vertex;
    let e2 = half_edges[is.he.prev].vertex - is.he.vertex;
    let in_normal = normalize(cross(e2, e1));
    let sv0 = simple_cubic_velocity(0);
    let in_vel = dot(local_ray, in_normal)/length(sv0);
    let ne1 = normalize(e1);
    let e2rej = e2 - dot(e2,ne1) * ne1;
    let ne2rej = normalize(e2rej);
    let side_vel = vec2f(dot(ne1, local_ray), dot(ne2rej, local_ray));
    let total_vel = vec3f(in_vel, side_vel); // cubic param first
    let global_surface_pos = is.tuv.y * e1 + is.tuv.z * e2;
    let local_pos = vec3f(0.0, dot(global_surface_pos, ne1), dot(global_surface_pos, ne2rej)); // cubic param first
    return HalfThroatTV(
        ht,
        TrianglePosState(is.he, local_triangle_from_halfedge(is.he), local_pos.yz),
        local_pos.x,
        total_vel
    );
}


@binding(0) @group(0) var<uniform> camera : Camera;
@binding(0) @group(1) var<storage> half_edges: array<HalfEdge>;

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