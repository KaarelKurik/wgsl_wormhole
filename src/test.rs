use std::f32::consts::{PI, TAU};

use cgmath::{Matrix3, Vector3};
use encase::ShaderType;

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct Camera {
    width: f32,
    height: f32,
    frame: Matrix3<f32>,
    frame_inv: Matrix3<f32>,
    centre: Vector3<f32>,
    yfov: f32,
    chart_index: u32,
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct Hermite {
    pos: Vector3<f32>,
    normal: Vector3<f32>,
}

// m >= 2
pub fn sphere(m: usize, n: usize) -> Vec<Hermite> {
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