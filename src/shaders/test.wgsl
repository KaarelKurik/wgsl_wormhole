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

@binding(0) @group(0) var<uniform> camera : Camera;
@binding(1) @group(0) var sampler0 : sampler;
@binding(0) @group(1) var skybox_array : texture_cube_array<f32>;


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