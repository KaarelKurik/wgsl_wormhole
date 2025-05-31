struct Camera {
    width: f32,
    height: f32,
    frame: mat3x3<f32>,
    frame_inv: mat3x3<f32>,
    centre: vec3<f32>,
    yfov: f32,
}

struct TR3 {
  q: vec3f,
  v: vec3f,
}

struct Hermite {
  pos: vec3f,
  normal: vec3f,
}

fn transform_bound_vec(tf: mat4x4<f32>, bv: vec3f) -> vec3f {
  return (tf * vec4f(bv, 1.0)).xyz;
}

fn transform_free_vec(tf: mat4x4<f32>, fv: vec3f) -> vec3f {
  return (tf * vec4f(fv, 0.0)).xyz;
}

fn transform_hermite(tf: mat4x4<f32>, h: Hermite) -> Hermite {
  return Hermite(transform_bound_vec(tf, h.pos), transform_free_vec(tf, h.normal));
}

fn transform_qv(tf: mat4x4<f32>, qv: TR3) -> TR3 {
  return TR3(transform_bound_vec(tf, qv.q), transform_free_vec(tf, qv.v));
}

struct SituatedPoint {
  chart_index: u32,
  q: vec3f,
}

struct SituatedTR3 {
  chart_index: u32,
  q: vec3f,
  v: vec3f,
}

struct SituatedTransform {
  chart_index: u32,
  local_to_global: mat4x4<f32>,
  global_to_local: mat4x4<f32>,
  global_to_local_norm: f32,
}

struct ThroatMetadata {
  transforms: array<SituatedTransform, 2>,
  half_throat_indices: array<u32, 2>,
  point_count: u32, // we assume >= 1
  support: f32,
  outer_length: f32,
}

struct HalfThroatData {
  throat_index: u32,
  side: u32, // 0 for a side, 1 for b side
}

fn wendland(h: f32, x: f32) -> f32 {
    if (x > h) {
        return 0.0;
    } else {
        var t = 1 - x / h;
        t = t * t;
        t = t * t;
        return t * (4 * x / h + 1);
    }
}

fn smootherstep(x: f32) -> f32 {
  if (x < 0) {
    return 0.0;
  }
  if (x > 1) {
    return 1.0;
  }
  return x * x * x * (10 + x * (-15 + x * 6));
}

struct CentroidRes {
  h: Hermite,
  near: bool,
}

// x throat local coords
fn local_centroid(kth_throat: u32, x: vec3f) -> CentroidRes {
  var weighted_normal_sum = vec3f(0);
  var weighted_pos_sum = vec3f(0);
  var weight_sum: f32 = 0;
  let throat_meta = throat_metas[kth_throat];
  for (var i: u32 = 0; i < throat_meta.point_count; i++) {
    let ith_hermite = index_throat_point(kth_throat, i);
    let w = wendland(throat_meta.support, length(x - ith_hermite.pos));
    weighted_normal_sum += ith_hermite.normal * w;
    weighted_pos_sum += ith_hermite.pos * w;
    weight_sum += w;
  }
  if (weight_sum <= 0) {
    return CentroidRes(
      Hermite(), // TODO: check if this works
      false,
    );
  }
  return CentroidRes(
    Hermite(weighted_pos_sum / weight_sum, weighted_normal_sum / weight_sum),
    true,
  );
}

// x throat local coords
fn distant_energy(kth_throat: u32, x: vec3f) -> f32 {
  let point_count = throat_metas[kth_throat].point_count;

  let zeroth_hermite = index_throat_point(kth_throat, 0);
  var mindist: f32 = length(x - zeroth_hermite.pos);

  var outenergy = dot(x-zeroth_hermite.pos, zeroth_hermite.normal);
  for (var i: u32 = 1; i < point_count; i++) {
    let ith_hermite = index_throat_point(kth_throat, i);
    let dist = length(x-ith_hermite.pos);
    if (dist < mindist) {
      outenergy = dot(x-ith_hermite.pos, ith_hermite.normal);
    }
  }
  return outenergy;
}

fn distant_energy_global_coords(half_throat_index: u32, gx: vec3f) -> f32 {
  let ht = half_throats[half_throat_index];
  let throat_index = ht.throat_index;
  let transform = throat_metas[throat_index].transforms[ht.side];
  let point_count = throat_metas[throat_index].point_count;
  let ltg = transform.local_to_global;

  let zeroth_local_hermite = index_throat_point(throat_index, 0);
  let zeroth_hermite = transform_hermite(ltg, zeroth_local_hermite);
  var mindist: f32 = length(gx - zeroth_hermite.pos);

  var outenergy = dot(gx-zeroth_hermite.pos, zeroth_hermite.normal);
  for (var i : u32 = 1; i < point_count; i++) {
    let ith_local_hermite = index_throat_point(throat_index, i);
    let ith_hermite = transform_hermite(ltg, ith_local_hermite);
    let dist = length(gx-ith_hermite.pos);
    if (dist < mindist) {
      outenergy = dot(gx-ith_hermite.pos, ith_hermite.normal);
    }
  }
  return outenergy;
}

// x throat local coords
fn surface_energy(kth_throat: u32, x: vec3f) -> f32 {
  let cr = local_centroid(kth_throat, x);
  if (cr.near) {
    return dot(cr.h.normal, x - cr.h.pos);
  }
  return distant_energy(kth_throat, x);
}

// NB! Not the same function as surface energy under a different chart.
// Reason why is that we need to know the "true" ambient distance
// to do the ray marching correctly.
fn surface_energy_global_coords(half_throat_index: u32, gx: vec3f) -> f32 {
  let ht = half_throats[half_throat_index];
  let throat_index = ht.throat_index;
  let transform = throat_metas[throat_index].transforms[ht.side];
  let gtl = transform.global_to_local;
  let ltg = transform.local_to_global;
  let lx = transform_bound_vec(gtl, gx);
  let lcr = local_centroid(throat_index, lx);
  if (lcr.near) {
    let gh = transform_hermite(ltg, lcr.h);
    return dot(gh.normal, gx - gh.pos);
  }
  return distant_energy_global_coords(throat_index, gx);
}

// Numerical. Might be risky to do it this way.
// Probably if any of our diffs should be done symbolically, it's this.
// x throat local coords
fn surface_energy_gradient(kth_throat: u32, x: vec3f, delta: f32) -> vec3f {
  let base = surface_energy(kth_throat, x);
  let sx = surface_energy(kth_throat, x + vec3f(delta,0,0));
  let sy = surface_energy(kth_throat, x + vec3f(0,delta,0));
  let sz = surface_energy(kth_throat, x + vec3f(0,0,delta));
  return vec3f(sx-base, sy-base, sz-base)/delta;
}

// x throat local coords
fn project_onto_surface(kth_throat: u32, x: vec3f, delta: f32) -> vec3f {
  const KSTEPS: u32 = 10;
  var lambda: f32 = 0;
  var base = x;
  var grad = surface_energy_gradient(kth_throat, x, delta);
  for (var k: u32 = 0; k < KSTEPS; k++) {
    base = x - lambda * grad;
    grad = surface_energy_gradient(kth_throat, base, delta); // could make cheaper by doing pushforward directly
    lambda += surface_energy(kth_throat, base)/dot(grad, grad); // might have to be dot(grad1, grad0) instead
  }
  return base;
}

// x throat local coords
fn base_and_delta(kth_throat: u32, x: vec3f, delta: f32) -> TR3 {
  let base = project_onto_surface(kth_throat, x, delta);
  return TR3(base, x-base);
}

fn transition_point(kth_throat: u32, x: vec3f, delta: f32) -> vec3f {
  let bd = base_and_delta(kth_throat, x, delta);
  let throat_meta = throat_metas[kth_throat];
  let new_delta = (throat_meta.outer_length - length(bd.v)) * normalize(bd.v);
  return bd.q + new_delta;
}

fn transition_jacobian(kth_throat: u32, x: vec3f, delta: f32) -> mat3x3f {
  let base = transition_point(kth_throat, x, delta);
  let sx = transition_point(kth_throat, x + vec3f(delta,0,0), delta);
  let sy = transition_point(kth_throat, x + vec3f(0,delta,0), delta);
  let sz = transition_point(kth_throat, x + vec3f(0,0,delta), delta);
  return transpose(mat3x3f(sx-base, sy-base, sz-base) * (1/delta));
}

// Suppose we have metric g on side A and metric h on side B. Coherence demands that
// the pullback of h by the A->B transition must be g. We're in a situation where g=h,
// so we're looking at pullback(g)=g. Since we define g = lambda*r + pullback(lambda)*pullback(r),
// and pullback composed with itself is identity (bc our transition is own inverse), we
// get pullback(g) = g with no issues.
// x local throat coords btw
fn metric(kth_throat: u32, x: vec3f, delta: f32) -> mat3x3f {
  let bd = base_and_delta(kth_throat, x, delta);
  let outer_length = throat_metas[kth_throat].outer_length;
  let other_bd = TR3(bd.q, (outer_length - length(bd.v)) * normalize(bd.v));
  let here_raw_fiber_param = length(bd.v) / outer_length;
  let there_raw_fiber_param = length(other_bd.v) / outer_length;
  let here_fiber_param = smootherstep(here_raw_fiber_param);
  let there_fiber_param = smootherstep(there_raw_fiber_param);
  let here_metric = mat3x3f(1,0,0,0,1,0,0,0,1);
  let jacobian = transition_jacobian(kth_throat, x, delta);
  let there_metric_pullback = transpose(jacobian) * jacobian;
  return here_fiber_param * here_metric + there_fiber_param * there_metric_pullback;
}

fn cofactor(x: mat3x3f) -> mat3x3f {
  return mat3x3f(
    x[1][1]*x[2][2]-x[2][1]*x[1][2], -x[1][0]*x[2][2]+x[2][0]*x[1][2], x[1][0]*x[2][1]-x[2][0]*x[1][1],
    -x[0][1]*x[2][2]+x[0][2]*x[2][1], x[0][0]*x[2][2]-x[0][2]*x[2][0], -x[0][0]*x[2][1]+x[0][1]*x[2][0],
    x[0][1]*x[1][2]-x[0][2]*x[1][1], -x[0][0]*x[1][2]+x[0][2]*x[1][0], x[0][0]*x[1][1]-x[0][1]*x[1][0]
  );
}

fn adjugate(x: mat3x3f) -> mat3x3f {
  return transpose(cofactor(x));
}

fn matrix_inverse(x: mat3x3f) -> mat3x3f {
  let adj = adjugate(x);
  let dm = x * adj;
  return adj * (1/dm[0][0]);
}

fn christoffel(kth_throat: u32, x: vec3f, delta: f32) -> array<mat3x3f, 3> {
  let idmat = mat3x3f(1,0,0,0,1,0,0,0,1);
  var cs: array<mat3x3f, 3>;
  let here_metric = metric(kth_throat, x, delta);
  for (var i: u32 = 0; i < 3; i++) {
    cs[i] = (metric(kth_throat, x + delta*idmat[i], delta) - here_metric) * (1/delta);
  }
  let inverse_metric = matrix_inverse(here_metric);
  var out: array<mat3x3f, 3>;
  for (var k: u32 = 0; k < 3; k++) {
    for (var i: u32 = 0; i < 3; i++) {
      for (var j: u32 = 0; j < 3; j++) {
        for (var m: u32 = 0; m < 3; m++) {
          out[k][i][j] = 0.5 * inverse_metric[k][m] * (cs[j][m][i] + cs[i][m][j] - cs[m][i][j]);
        }
      }
    }
  }
  return out;
}

fn accel_here(kth_throat: u32, qv: TR3, delta: f32) -> vec3f {
  let c = christoffel(kth_throat, qv.q, delta);
  var out: vec3f;
  out[0] = -dot(qv.v, c[0] * qv.v);
  out[1] = -dot(qv.v, c[1] * qv.v);
  out[2] = -dot(qv.v, c[2] * qv.v);
  return out;
}

fn phase_vel(kth_throat: u32, qv: TR3, delta: f32) -> TR3 {
  return TR3(qv.v, accel_here(kth_throat, qv, delta));
}

fn one_outer_step(kth_throat: u32, qv: TR3) -> TR3 {
  let energy = surface_energy(kth_throat, qv.q);
  let outer_length = throat_metas[kth_throat].outer_length;
  let delta = energy - outer_length;
  if (delta <= 0.2 * outer_length) {
    return qv;
  }
  return TR3(qv.q + delta * qv.v, qv.v);
}

struct StepBoundRes {
  bound: f32,
  near: bool,
}

fn step_length_bound(kth_throat: u32, q: vec3f) -> StepBoundRes {
  let energy = surface_energy(kth_throat, q);
  let outer_length = throat_metas[kth_throat].outer_length;
  let delta = energy - outer_length;
  let near = delta <= 0.2 * outer_length;
  return StepBoundRes(delta, near);
}

struct OuterStepRes {
  min_energy_half_throat_index: u32,
  close_to_throat: bool,
  qv: TR3,
}

fn one_outer_step_all_global(nth_chart: u32, gqv: TR3) -> OuterStepRes {
  var step_length = 20.0; // max step length
  var min_index: u32 = 0;
  for (var i: u32 = half_throat_range_start_by_chart_index[nth_chart]; i < half_throat_range_end_by_chart_index[nth_chart]; i++) {
    let ht = half_throats[i];
    let tm = throat_metas[ht.throat_index];
    let gtl = tm.transforms[ht.side].global_to_local;
    let gtl_norm = tm.transforms[ht.side].global_to_local_norm;
    let lqv = transform_qv(gtl, gqv);
    let boundres = step_length_bound(ht.throat_index, lqv.q);
    if (boundres.near) {
      return OuterStepRes(i, true, gqv);
    }
    let br = boundres.bound / gtl_norm;
    if (br < step_length) {
      step_length = br;
      min_index = i;
    }
  }
  return OuterStepRes(min_index, false, TR3(gqv.q + step_length * normalize(gqv.v), gqv.v));
}

fn one_throat_step(kth_throat: u32, qv: TR3, dt:f32, delta: f32) -> TR3 {
  let k1 = phase_vel(kth_throat, qv, delta);
  let x1 = TR3(qv.q + (dt/2)*k1.q, qv.v + (dt/2)*k1.v);
  let k2 = phase_vel(kth_throat, x1, delta);
  let x2 = TR3(qv.q + (dt/2)*k2.q, qv.v + (dt/2)*k2.v);
  let k3 = phase_vel(kth_throat, x2, delta);
  let x3 = TR3(qv.q + dt*k3.q, qv.v + dt*k3.v);
  let k4 = phase_vel(kth_throat, x3, delta);
  return TR3(qv.q + (dt/6)*(k1.q + 2*k2.q + 2*k3.q + k4.q), qv.v + (dt/6)*(k1.v + 2*k2.v + 2*k3.v + k4.v));
}

fn ambient_step(aqv: SituatedTR3) -> SituatedTR3 {
  let ambient_step_res = one_outer_step_all_global(aqv.chart_index, TR3(aqv.q, aqv.v));
  if (ambient_step_res.close_to_throat) {
    let half_throat_index = ambient_step_res.min_energy_half_throat_index;
    let gtl = global_to_local(half_throat_index);
    let transformed_qv = transform_qv(gtl, ambient_step_res.qv);
    let chart_index = half_throat_index_to_chart_index(half_throat_index);
    return SituatedTR3(chart_index, transformed_qv.q, transformed_qv.v);
  }
  return SituatedTR3(aqv.chart_index, ambient_step_res.qv.q, ambient_step_res.qv.v);
}

fn throat_side_transition(tq: SituatedPoint, diff_scale: f32) -> SituatedPoint {
  let half_throat_index = chart_index_to_half_throat_index(tq.chart_index);
  let op_index = opposite_half_throat_index(half_throat_index);
  let op_chart_index = half_throat_index_to_chart_index(op_index);
  let half_throat = half_throats[half_throat_index];
  let throat_index = half_throat.throat_index;
  let outer_length = throat_metas[throat_index].outer_length;
  let bd = base_and_delta(throat_index, tq.q, diff_scale);
  let new_v = (outer_length - length(bd.v)) * normalize(bd.v);
  let new_pos = bd.q + new_v;
  return SituatedPoint(op_chart_index, new_pos);
}

fn throat_side_transition_pushforward(tqv: SituatedTR3, diff_scale: f32) -> SituatedTR3 {
  let tq = SituatedPoint(tqv.chart_index, tqv.q);
  let scale_factor = length(tqv.v)/diff_scale;
  let tqd = SituatedPoint(tqv.chart_index, tqv.q + diff_scale * normalize(tqv.v));
  let fv = throat_side_transition(tq, diff_scale);
  let fvdv = throat_side_transition(tqd, diff_scale);
  let new_chart_index = fv.chart_index;
  let new_v = scale_factor * (fvdv.q - fv.q);
  return SituatedTR3(new_chart_index, fv.q, new_v);
}

fn throat_step(tqv: SituatedTR3, diff_scale: f32, time_scale: f32) -> SituatedTR3 {
  let half_throat_index = chart_index_to_half_throat_index(tqv.chart_index);
  let half_throat = half_throats[half_throat_index];
  let throat_index = half_throat.throat_index;
  let simple_throat_step_res = one_throat_step(throat_index, TR3(tqv.q, tqv.v), time_scale, diff_scale);
  let new_base_and_delta = base_and_delta(throat_index, simple_throat_step_res.q, diff_scale);
  let outer_length = throat_metas[throat_index].outer_length;
  if (length(new_base_and_delta.v) < 0.3 * outer_length) { // magic number
    return throat_side_transition_pushforward(SituatedTR3(tqv.chart_index, simple_throat_step_res.q, simple_throat_step_res.v), diff_scale);
  }
  if (length(new_base_and_delta.v) >= 1.2 * outer_length) { // synced with 0.2 in step_length_bound 
    // transform simple step res to ambient
    let situated_transform = throat_metas[throat_index].transforms[half_throat.side];
    let gqv = transform_qv(situated_transform.local_to_global, TR3(simple_throat_step_res.q, simple_throat_step_res.v));
    let ambient_index = situated_transform.chart_index;
    return SituatedTR3(ambient_index, gqv.q, gqv.v);
  }
  return SituatedTR3(tqv.chart_index, simple_throat_step_res.q, simple_throat_step_res.v);
}

fn one_general_step(qv: SituatedTR3, diff_scale: f32, time_scale: f32) -> SituatedTR3 {
  if (qv_is_in_ambient(qv)) {
    return ambient_step(qv);
  }
  return throat_step(qv, diff_scale, time_scale);
}

fn transform_ambient_to_half_throat(half_throat_index: u32, aqv: SituatedTR3) -> SituatedTR3 {
  let half_throat = half_throats[half_throat_index];
}

fn local_to_global(half_throat_index: u32) -> mat4x4f {
  let half_throat = half_throats[half_throat_index];
  let transforms = throat_metas[half_throat.throat_index].transforms[half_throat.side];
  return transforms.local_to_global;
}

fn global_to_local(half_throat_index: u32) -> mat4x4f {
  let half_throat = half_throats[half_throat_index];
  let transforms = throat_metas[half_throat.throat_index].transforms[half_throat.side];
  return transforms.global_to_local;
}

fn count_ambient_chart_indices() -> u32 {
  return arrayLength(&half_throat_range_end_by_chart_index);
}

fn chart_is_ambient(chart_index: u32) -> bool {
  return chart_index < count_ambient_chart_indices();
}

fn chart_index_to_half_throat_index(chart_index: u32) -> u32 {
  return chart_index - count_ambient_chart_indices();
}

fn half_throat_index_to_chart_index(half_throat_index: u32) -> u32 {
  return half_throat_index + count_ambient_chart_indices();
}

fn opposite_half_throat_index(half_throat_index: u32) -> u32 {
  let ht = half_throats[half_throat_index];
  let throat_index = ht.throat_index;
  let tm = throat_metas[throat_index];
  return tm.half_throat_indices[1-ht.side];
}

fn qv_is_in_ambient(qv: SituatedTR3) -> bool {
  return chart_is_ambient(qv.chart_index);
}

@binding(0) @group(0) var<uniform> camera : Camera;
@binding(1) @group(0) var sampler0 : sampler;

@binding(0) @group(1) var skybox_array : texture_cube_array<f32>;

@binding(0) @group(2) var<storage> throat_metas: array<ThroatMetadata>;
@binding(1) @group(2) var<storage> throat_point_starts: array<u32>;
@binding(2) @group(2) var<storage> throat_local_points: array<Hermite>;
@binding(3) @group(2) var<storage> half_throats: array<HalfThroatData>;
@binding(4) @group(2) var<storage> half_throat_range_start_by_chart_index: array<u32>;
@binding(5) @group(2) var<storage> half_throat_range_end_by_chart_index: array<u32>;

fn index_throat_point(kth_throat: u32, nth_point: u32) -> Hermite {
  return throat_local_points[throat_point_starts[kth_throat] + nth_point];
}

fn fragpos_to_ray(camera: Camera, pos: vec2f)->TR3 {
    let ray_coords = normalize(vec3f( // Note the normalization - if camera frame is orthonormal, ray will be also
        (pos.x / camera.width - 0.5) * (camera.width / camera.height),
        pos.y / camera.height - 0.5,
        0.5 / (tan(camera.yfov / 2.0))
    ));
    let ray = camera.frame * ray_coords;
    return TR3(camera.centre, ray);
}

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
  let color = textureSample(skybox_array, sampler0, ray.v, 1);
  return color;
}