use cgmath::{Matrix3, Matrix4, Vector3};
use encase::ShaderType;

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct Camera {
    pub width: f32,
    pub height: f32,
    pub frame: Matrix3<f32>,
    pub frame_inv: Matrix3<f32>,
    pub centre: Vector3<f32>,
    pub yfov: f32,
    pub chart_index: [u32; 3],
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub(crate) struct HalfEdge {
    pub(crate) index: u32,
    pub(crate) prev: u32,
    pub(crate) next: u32,
    pub(crate) twin: u32,
    pub(crate) vertex: Vector3<f32>,
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub(crate) struct Mesh {
    pub(crate) he_lo_index: u32,
    pub(crate) he_hi_index: u32,
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct HalfThroat {
    pub(crate) ltg: Matrix4<f32>,
    pub(crate) gtl: Matrix4<f32>,
    pub(crate) index: u32,
    pub(crate) ambient_index: u32,
    pub(crate) twin_index: u32,
    pub(crate) mesh: Mesh,
    pub(crate) mid_t: f32,           // other side must have same mid_t
    pub(crate) hi_t: f32,            // 2*mid_t - 1 > hi_t > mid_t
    pub(crate) throat_to_mid_t: f32, // mid_t > throat_to_mid_t > mid_to_throat_t > 1
    pub(crate) mid_to_throat_t: f32,
}

pub(crate) fn tetrahedron() -> Vec<HalfEdge> {
    let verts = [
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(1.0, -1.0, -1.0),
        Vector3::new(-1.0, 1.0, -1.0),
        Vector3::new(-1.0, -1.0, 1.0),
    ];
    let faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];

    let mut half_edges = Vec::with_capacity(12);

    // Create half-edges for each face
    for face_idx in 0..4 {
        let face = faces[face_idx];
        let base_he_idx = face_idx * 3;

        for i in 0..3 {
            let he_idx = base_he_idx + i;
            let next_idx = base_he_idx + ((i + 1) % 3);
            let prev_idx = base_he_idx + ((i + 2) % 3);

            half_edges.push(HalfEdge {
                index: he_idx as u32,
                prev: prev_idx as u32,
                next: next_idx as u32,
                twin: 0, // Will be set below
                vertex: verts[face[i]],
            });
        }
    }

    // Set up twin relationships
    let twins = [
        (0,5),
        (1,11),
        (2,6),
        (3,8),
        (4,9),
        (5,0),
        (6,2),
        (7,10),
        (8,3),
        (9,4),
        (10,7),
        (11,1)
    ];

    for (he_idx, twin_idx) in twins {
        half_edges[he_idx].twin = twin_idx as u32;
    }

    half_edges
}
