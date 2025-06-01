use std::{
    array,
    f32::consts::{PI, TAU},
    num::NonZero,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use cgmath::{Array, InnerSpace, Matrix3, Matrix4, Rad, SquareMatrix, Vector3, Zero};
use encase::{ShaderType, StorageBuffer, UniformBuffer, internal::WriteInto};
use image::{EncodableLayout, ImageBuffer, ImageError, RgbaImage, imageops::FilterType};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{self, DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop,
    keyboard::{KeyCode, PhysicalKey},
    window::{self, CursorGrabMode, Window},
};

struct Keeper<T> {
    val: T,
    group: u32,
    binding: u32,
    changed: bool,
}

impl<T> Keeper<T> {
    fn peek_val(&self) -> &T {
        &self.val
    }

    fn get_val(&mut self) -> &mut T {
        self.changed = true;
        &mut self.val
    }

    fn group(&self) -> u32 {
        self.group
    }

    fn binding(&self) -> u32 {
        self.binding
    }

    fn reset(&mut self) {
        self.changed = false;
    }
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct Camera {
    width: f32,
    height: f32,
    frame: Matrix3<f32>,
    frame_inv: Matrix3<f32>,
    centre: Vector3<f32>,
    yfov: f32,
    chart_index: u32,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct TR3 {
    q: Vector3<f32>,
    v: Vector3<f32>,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct Hermite {
    pos: Vector3<f32>,
    normal: Vector3<f32>,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct SituatedPoint {
    chart_index: u32,
    q: Vector3<f32>,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct SituatedTR3 {
    chart_index: u32,
    q: Vector3<f32>,
    v: Vector3<f32>,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct SituatedTransform {
    chart_index: u32,
    local_to_global: Matrix4<f32>,
    global_to_local: Matrix4<f32>,
    global_to_local_norm: f32,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct ThroatMetadata {
    transforms: [SituatedTransform; 2],
    half_throat_indices: [u32; 2],
    point_count: u32,
    support: f32,
    outer_length: f32,
}

#[derive(Debug, Clone, Copy, ShaderType)]
struct HalfThroatData {
    throat_index: u32,
    side: u32,
}

#[derive(Debug, Clone, Copy)]
struct CameraController {
    q_state: ElementState,
    e_state: ElementState,
    w_state: ElementState,
    s_state: ElementState,
    a_state: ElementState,
    d_state: ElementState,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            q_state: ElementState::Released,
            e_state: ElementState::Released,
            w_state: ElementState::Released,
            s_state: ElementState::Released,
            a_state: ElementState::Released,
            d_state: ElementState::Released,
        }
    }
}

impl CameraController {
    // TODO: modify this to use the metric
    fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        const ANGULAR_SPEED: f32 = 1f32;
        const LINEAR_SPEED: f32 = 8f32;
        let dt_seconds = dt.as_secs_f32();

        let mut linvel = Vector3::<f32>::zero();
        let z_linvel = LINEAR_SPEED * Vector3::unit_z();
        let x_linvel = LINEAR_SPEED * Vector3::unit_x();

        if self.w_state.is_pressed() {
            linvel += z_linvel;
        }
        if self.s_state.is_pressed() {
            linvel -= z_linvel;
        }
        if self.d_state.is_pressed() {
            linvel += x_linvel;
        }
        if self.a_state.is_pressed() {
            linvel -= x_linvel;
        }
        camera.centre += camera.frame * (dt_seconds * linvel);

        let mut rotvel = Vector3::<f32>::zero();
        let z_rotvel = ANGULAR_SPEED * Vector3::unit_z();

        if self.q_state.is_pressed() {
            rotvel -= z_rotvel;
        }
        if self.e_state.is_pressed() {
            rotvel += z_rotvel;
        }
        let axis = rotvel.normalize();
        if axis.is_finite() {
            camera.frame =
                camera.frame * Matrix3::from_axis_angle(axis, Rad(dt_seconds * rotvel.magnitude()));
        }
        camera.frame_inv = camera.frame.invert().unwrap();
    }
    fn process_window_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        repeat: false,
                        state,
                        ..
                    },
                ..
            } => match code {
                // TODO: refactor this to have a single source of truth
                KeyCode::KeyQ => self.q_state = *state,
                KeyCode::KeyE => self.e_state = *state,
                KeyCode::KeyW => self.w_state = *state,
                KeyCode::KeyS => self.s_state = *state,
                KeyCode::KeyA => self.a_state = *state,
                KeyCode::KeyD => self.d_state = *state,
                _ => {}
            },
            _ => {}
        }
    }
    fn process_mouse_motion(&mut self, camera: &mut Camera, delta: &(f64, f64)) {
        const ANGULAR_SPEED: f32 = 0.001;
        let dx = delta.0 as f32;
        let dy = delta.1 as f32;
        let angle = ANGULAR_SPEED * (dx.powi(2) + dy.powi(2)).sqrt();
        let axis = Vector3 {
            x: -dy,
            y: dx,
            z: 0.0,
        }
        .normalize();
        camera.frame = camera.frame * Matrix3::from_axis_angle(axis, Rad(angle));
    }
}

// m >= 2
fn sphere(m: usize, n: usize) -> Vec<Hermite> {
    let mut out = Vec::with_capacity((m - 2) * n + 2);
    out.push(Hermite {
        pos: Vector3::unit_z(),
        normal: Vector3::unit_z(),
    });
    for i in 1..(m - 1) {
        let psi = (i as f32 / (m - 1) as f32) * PI;
        for j in 0..n {
            let phi = (j as f32 / n as f32) * TAU;
            let v = Vector3::new(phi.cos() * psi.sin(), phi.sin() * psi.sin(), psi.cos());
            out.push(Hermite { pos: v, normal: v });
        }
    }
    out.push(Hermite {
        pos: -Vector3::unit_z(),
        normal: -Vector3::unit_z(),
    });
    out
}

fn interim_storage_buffer<T: ShaderType + WriteInto>(st: &T) -> StorageBuffer<Vec<u8>> {
    let mut o = StorageBuffer::new(Vec::<u8>::new());
    o.write(st);
    o
}

// Has to have equal resolution on all faces
struct RgbaSkybox {
    px: RgbaImage,
    nx: RgbaImage,
    py: RgbaImage,
    ny: RgbaImage,
    pz: RgbaImage,
    nz: RgbaImage,
}

impl RgbaSkybox {
    fn dimensions(&self) -> (u32, u32) {
        self.px.dimensions()
    }
    fn width(&self) -> u32 {
        self.px.width()
    }
    fn height(&self) -> u32 {
        self.px.height()
    }
    fn extent(&self) -> Extent3d {
        Extent3d {
            width: self.width(),
            height: self.height(),
            depth_or_array_layers: 6,
        }
    }
    fn texture_format(&self) -> TextureFormat {
        TextureFormat::Rgba8UnormSrgb
    }
    fn descriptor(&self) -> TextureDescriptor {
        TextureDescriptor {
            label: None,
            size: self.extent(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: self.texture_format(),
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }
    }
    fn view_descriptor(&self) -> TextureViewDescriptor {
        TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            format: Some(self.texture_format()),
            ..Default::default()
        }
    }
    fn as_bytevec(&self) -> Vec<u8> {
        [
            self.px.as_bytes(),
            self.nx.as_bytes(),
            self.py.as_bytes(),
            self.ny.as_bytes(),
            self.pz.as_bytes(),
            self.nz.as_bytes(),
        ]
        .concat()
    }
    fn bytes_per_block(&self) -> u32 {
        4
    }
    fn load_from_path(bg_path: &Path) -> Result<Self, ImageError> {
        let [px, nx, py, ny, pz, nz]: [Result<_, ImageError>; 6] =
            ["right", "left", "bottom", "top", "front", "back"].map(|x| {
                let mut im = image::open(bg_path.join(format!("{}.png", x)))?.into_rgba8();
                image::imageops::flip_vertical_in_place(&mut im);
                Ok(im)
            });
        Ok(RgbaSkybox {
            px: px?,
            nx: nx?,
            py: py?,
            ny: ny?,
            pz: pz?,
            nz: nz?,
        })
    }
    fn resize(&self, nwidth: u32, nheight: u32, filter: FilterType) -> RgbaSkybox {
        RgbaSkybox {
            px: image::imageops::resize(&self.px, nwidth, nheight, filter),
            nx: image::imageops::resize(&self.nx, nwidth, nheight, filter),
            py: image::imageops::resize(&self.py, nwidth, nheight, filter),
            ny: image::imageops::resize(&self.ny, nwidth, nheight, filter),
            pz: image::imageops::resize(&self.pz, nwidth, nheight, filter),
            nz: image::imageops::resize(&self.nz, nwidth, nheight, filter),
        }
    }
}

struct SkyboxArray {
    dimensions: (u32, u32),
    skyboxes: Vec<RgbaSkybox>,
}

impl SkyboxArray {
    fn width(&self) -> u32 {
        self.dimensions.0
    }
    fn height(&self) -> u32 {
        self.dimensions.1
    }
    fn extent(&self) -> Extent3d {
        Extent3d {
            width: self.width(),
            height: self.height(),
            depth_or_array_layers: 6 * self.skyboxes.len() as u32,
        }
    }
    fn texture_format(&self) -> TextureFormat {
        TextureFormat::Rgba8UnormSrgb
    }
    fn descriptor(&self) -> TextureDescriptor {
        TextureDescriptor {
            label: None,
            size: self.extent(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: self.texture_format(),
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }
    }
    fn view_descriptor(&self) -> TextureViewDescriptor {
        TextureViewDescriptor {
            dimension: Some(TextureViewDimension::CubeArray),
            format: Some(self.texture_format()),
            ..Default::default()
        }
    }
    fn as_bytevec(&self) -> Vec<u8> {
        self.skyboxes
            .iter()
            .map(|q| q.as_bytevec())
            .collect::<Vec<_>>()
            .concat()
    }
    fn new(skyboxes: &[RgbaSkybox], width: u32, height: u32) -> Self {
        Self {
            dimensions: (width, height),
            skyboxes: skyboxes
                .iter()
                .map(|sb| sb.resize(width, height, FilterType::Nearest))
                .collect(),
        }
    }
    fn bytes_per_block(&self) -> u32 {
        4
    }
    fn write_texture(&self, texture: &Texture, queue: &Queue) {
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                aspect: TextureAspect::All,
            },
            &self.as_bytevec(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.bytes_per_block() * self.width()),
                rows_per_image: Some(self.height()),
            },
            self.extent(),
        );
    }
}

pub enum AppState<'a> {
    Uninitialized(),
    Initialized(App<'a>),
}

pub struct App<'a> {
    window: Arc<Window>,
    size: PhysicalSize<u32>,
    surface: Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Device,
    queue: Queue,
    mouse_capture_mode: CursorGrabMode,
    cursor_is_visible: bool,
    camera_controller: CameraController,
    camera: Camera,
    render_pipeline: RenderPipeline,
    bg0: BindGroup,
    fixed_time: Instant,
    camera_buffer: Buffer,
    skyboxes_bind_group: BindGroup,
    geometry_bind_group: BindGroup,
}

impl<'a> App<'a> {
    fn render(&mut self) -> Result<(), SurfaceError> {
        self.sync_logic_to_gpu();
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, Some(&self.bg0), &[]);
            render_pass.set_bind_group(1, Some(&self.skyboxes_bind_group), &[]);
            render_pass.set_bind_group(2, Some(&self.geometry_bind_group), &[]);
            render_pass.draw(0..3, 0..1);
        }
        let index = self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(MaintainBase::WaitForSubmissionIndex(index)).unwrap();
        output.present();

        Ok(())
    }
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.camera.height = new_size.height as f32;
            self.camera.width = new_size.width as f32;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }
    fn window_input(&mut self, event: &WindowEvent) {
        self.camera_controller.process_window_event(event);
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.toggle_mouse_capture(),
            _ => {}
        }
    }
    fn toggle_mouse_capture(&mut self) {
        let new_mode = match self.mouse_capture_mode {
            CursorGrabMode::None => CursorGrabMode::Locked,
            CursorGrabMode::Confined => CursorGrabMode::None,
            CursorGrabMode::Locked => CursorGrabMode::None,
        };
        let fallback_mode = match self.mouse_capture_mode {
            CursorGrabMode::None => CursorGrabMode::Confined,
            CursorGrabMode::Confined => CursorGrabMode::None,
            CursorGrabMode::Locked => CursorGrabMode::None,
        };
        let visibility = match new_mode {
            CursorGrabMode::None => true,
            CursorGrabMode::Confined => false,
            CursorGrabMode::Locked => false,
        };
        if let Err(_) = self.window.set_cursor_grab(new_mode) {
            self.window.set_cursor_grab(fallback_mode).unwrap();
        }
        self.window.set_cursor_visible(visibility);

        self.mouse_capture_mode = new_mode;
        self.cursor_is_visible = visibility;
    }
    fn device_input(&mut self, event: &DeviceEvent) {
        match self.mouse_capture_mode {
            CursorGrabMode::Confined | CursorGrabMode::Locked => {
                if let DeviceEvent::MouseMotion { delta } = event {
                    self.camera_controller
                        .process_mouse_motion(&mut self.camera, delta);
                }
            }
            _ => {}
        }
    }
    fn update_logic_state(&mut self, new_time: Instant) {
        let dt = new_time.duration_since(self.fixed_time);
        self.camera_controller.update_camera(&mut self.camera, dt);

        // Write all deferrable logic (not rendering) changes.
        // Maybe I should have wrapper logic just to set a bit telling me whether
        // a change needs to be written?
        self.fixed_time = new_time;
    }
    fn sync_logic_to_gpu(&mut self) {
        let mut interim_buffer = UniformBuffer::new(Vec::<u8>::new());
        interim_buffer.write(&self.camera).unwrap();
        self.queue
            .write_buffer(&self.camera_buffer, 0, &interim_buffer.into_inner());
    }
}

impl<'a> ApplicationHandler for AppState<'a> {
    fn new_events(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(app) => {
                let new_time = Instant::now();
                app.update_logic_state(new_time);
                if cause == winit::event::StartCause::Poll {
                    app.window.request_redraw();
                }
            }
        }
    }
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let AppState::Initialized(_) = self {
            panic!("Tried to initialize already-initialized app!");
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let size = window.as_ref().inner_size();

        let instance = wgpu::Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter_future = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        });
        let adapter = pollster::block_on(adapter_future).unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let supported_presentation_modes = surface_caps.present_modes;

        let mode_comparator = |pres_mode: &&wgpu::PresentMode| match pres_mode {
            wgpu::PresentMode::Immediate => -1, // my machine freezes every few secs with vsync now - not sure why
            wgpu::PresentMode::Mailbox => 0,
            wgpu::PresentMode::FifoRelaxed => 1,
            wgpu::PresentMode::Fifo => 2,
            _ => 3,
        };
        let present_mode = *supported_presentation_modes
            .iter()
            .min_by_key(mode_comparator)
            .unwrap();

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        let device_future = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | Features::TEXTURE_BINDING_ARRAY,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 4,
                max_bind_groups: 5,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        });
        let (device, queue) = pollster::block_on(device_future).unwrap();

        surface.configure(&device, &surface_config); // causes segfault if device, surface_config die.

        let shader_module = device.create_shader_module(include_wgsl!("shaders/test.wgsl"));

        let camera = Camera {
            width: size.width as f32,
            height: size.height as f32,
            frame: Matrix3::identity(),
            frame_inv: Matrix3::identity(),
            centre: Vector3::new(0.0, 0.0, -5.0),
            yfov: PI / 2.0,
            chart_index: 0,
        };

        let mut camera_interim_buffer = UniformBuffer::new(Vec::<u8>::new());
        camera_interim_buffer.write(&camera).unwrap();
        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: &camera_interim_buffer.into_inner(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let skybox_names = ["bg0", "bg_debug"];
        let skyboxes: Vec<RgbaSkybox> = skybox_names
            .iter()
            .map(|p| RgbaSkybox::load_from_path(Path::new(&format!("textures/{}", p))).unwrap())
            .collect();
        let skybox_array = SkyboxArray::new(&skyboxes, 1024, 1024);
        let skybox_array_texture = device.create_texture(&skybox_array.descriptor());
        skybox_array.write_texture(&skybox_array_texture, &queue);
        let skybox_array_texture_view =
            skybox_array_texture.create_view(&skybox_array.view_descriptor());
        let skybox_sampler = device.create_sampler(&SamplerDescriptor::default());

        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bg0_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg0"),
            layout: &bg0_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &camera_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&skybox_sampler),
                },
            ],
        });

        let skyboxes_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("skyboxes_bind_group_layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: Some(NonZero::new(skybox_array.skyboxes.len() as u32).unwrap()),
                }],
            });

        let skyboxes_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("skyboxes_bind_group"),
            layout: &skyboxes_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&skybox_array_texture_view),
            }],
        });

        let tf0 = SituatedTransform {
            chart_index: 0,
            local_to_global: Matrix4::identity(),
            global_to_local: Matrix4::identity(),
            global_to_local_norm: 1.0,
        };

        let tf1 = SituatedTransform {
            chart_index: 1,
            local_to_global: Matrix4::identity(),
            global_to_local: Matrix4::identity(),
            global_to_local_norm: 1.0,
        };

        let throat_points = sphere(40, 40);

        let main_throat = ThroatMetadata {
            transforms: [tf0, tf1],
            half_throat_indices: [0, 1],
            point_count: throat_points.len() as u32,
            support: 0.3,
            outer_length: 0.3,
        };

        let ht0 = HalfThroatData {
            throat_index: 0,
            side: 0,
        };

        let ht1 = HalfThroatData {
            throat_index: 0,
            side: 1,
        };

        let throat_metas = vec![main_throat];
        let throat_point_starts = vec![0u32];
        let throat_local_points = throat_points;
        let half_throats = vec![ht0, ht1];
        let half_throat_range_start_by_chart_index = vec![0u32, 1];
        let half_throat_range_end_by_chart_index = vec![1u32, 2];

        let throat_metas_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("throat_metas_buffer"),
            contents: &interim_storage_buffer(&throat_metas).into_inner(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let throat_point_starts_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("throat_point_starts_buffer"),
            contents: &interim_storage_buffer(&throat_point_starts).into_inner(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let throat_local_points_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("throat_local_points_buffer"),
            contents: &interim_storage_buffer(&throat_local_points).into_inner(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let half_throats_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("half_throats_buffer"),
            contents: &interim_storage_buffer(&half_throats).into_inner(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let half_throat_range_start_by_chart_index_buffer =
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("half_throat_range_start_by_chart_index_buffer"),
                contents: &interim_storage_buffer(&half_throat_range_start_by_chart_index)
                    .into_inner(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
        let half_throat_range_end_by_chart_index_buffer =
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("half_throat_range_end_by_chart_index_buffer"),
                contents: &interim_storage_buffer(&half_throat_range_end_by_chart_index)
                    .into_inner(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let geometry_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("geometry_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let geometry_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("geometry_bind_group"),
            layout: &geometry_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &throat_metas_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &throat_point_starts_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &throat_local_points_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &half_throats_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &half_throat_range_start_by_chart_index_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &half_throat_range_end_by_chart_index_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[
                &bg0_layout,
                &skyboxes_bind_group_layout,
                &geometry_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: Some("vtx_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: Some("frag_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let fixed_time = Instant::now();

        *self = AppState::Initialized(App {
            window,
            surface,
            size,
            surface_config,
            device,
            queue,
            mouse_capture_mode: CursorGrabMode::None,
            cursor_is_visible: true,
            camera,
            camera_controller: Default::default(),
            camera_buffer,
            render_pipeline,
            bg0,
            skyboxes_bind_group,
            geometry_bind_group,
            fixed_time,
        })
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: window::WindowId,
        event: event::WindowEvent,
    ) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(app) => match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => match app.render() {
                    Ok(_) => {}
                    Err(SurfaceError::Lost) => app.resize(app.size),
                    Err(SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                },
                WindowEvent::Resized(new_size) => app.resize(new_size),
                e => app.window_input(&e),
            },
        }
    }
    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match self {
            AppState::Uninitialized() => {}
            AppState::Initialized(app) => app.device_input(&event),
        }
    }
}
