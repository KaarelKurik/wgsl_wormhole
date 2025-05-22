use wgsl_wormhole::AppState;
use winit::event_loop::{self, EventLoop};

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    let mut app_state = AppState::Uninitialized();
    event_loop.run_app(&mut app_state).unwrap();
}
