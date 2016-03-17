extern crate cgmath;
#[macro_use]
extern crate glium;
extern crate time;

use cgmath::{Angle, BaseFloat, BaseNum, EuclideanVector, One, Quaternion, Rad, Rotation, Rotation3, Point, Point2, Point3, Vector, Vector2, Vector3, Zero};

use std::f32;
use time::{PreciseTime};

const SIZE: usize = 16;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
enum Direction {
    North = 0,
    South = 1,
    East = 2,
    West = 3,
}

/*#[derive(Clone, Copy)]
enum Wall {
    Permeable,
    Barrier,
}

#[derive(Clone, Copy)]
enum Item {
    Empty,
    // Wall,
    Pellet,
}

#[derive(Clone, Copy)]
struct Cell {
    walls: [ Wall ; 4 ],
    /*north: Wall,
    south: Wall,
    east: Wall,
    west: Wall,*/
    item: Item,
}*/

#[derive(Clone, Copy, Debug)]
struct Face {
    /// point on the plane for this face
    position: Vector3<f32>,
    /// normal vector for the plane of this face
    normal: Vector3<f32>,
}


type Board = [ Face ; 6 ];

#[derive(Clone, Copy, Debug)]
struct Mob {
    /// current face for this mob
    face: usize,
    /// current position on this face
    position: Vector3<f32>,
    /// current velocity for this mob
    velocity: Vector3<f32>,
}

impl Mob {
    /// Move.
    fn physics(&mut self, board: &Board, t: f32) {
        let delta = self.velocity * t;
        let l = self.position + delta;
        let mut nearest_face = (self.face, 1.0);

        for (face, &Face { position, normal }) in board.into_iter().enumerate() {
            let denom = l.dot(normal);
            if denom.is_normal() {
                let distance = (position.dot(normal)) / denom;
                if distance.is_sign_positive() && (distance < nearest_face.1/* || distance == nearest_face.1 && face != self.face*/) {
                    nearest_face = (face, distance);
                }
            }
        }

        let q = Quaternion::between_vectors(board[self.face].normal, board[nearest_face.0].normal);
        self.velocity = q.rotate_vector(self.velocity);
        self.face = nearest_face.0;
        fn clamp(n: f32) -> f32 {
            if n < -(SIZE as f32) {
                -(SIZE as f32)
            } else if n > SIZE as f32 {
                SIZE as f32
            } else { n }
        }
        // let p = d * l; // point of intersection
        // There's leftover distance to travel: specifically, the remaining delta.
        //let Vector3 { x, y, z } = l + self.velocity * t;
        //let Vector3 { x, y, z } = l + self.velocity * t * / l.magnitude();
        let Vector3 { x, y, z } = l + self.velocity * t;
        self.position = Vector3 { x: clamp(x), y: clamp(y), z: clamp(z) };
    }
}

#[derive(Clone, Copy, Debug)]
struct Ghost {
    mob: Mob,
}

#[derive(Clone, Copy, Debug)]
struct Player {
    mob: Mob,
}

#[derive(Clone, Copy, Debug)]
struct Game {
    board: Board,
    ghosts: [ Ghost ; 4 ],
    player: Player,
}

#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

implement_vertex!(Vertex, position, normal);

#[derive(Copy, Clone, Debug)]
struct FaceVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
}

implement_vertex!(FaceVertex, position, normal, tex_coords);

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
             up[2] * f[0] - up[0] * f[2],
             up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
             f[2] * s_norm[0] - f[0] * s_norm[2],
             f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
             -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
             -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s[0], u[0], f[0], 0.0],
        [s[1], u[1], f[1], 0.0],
        [s[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

fn main() {
    // TODO: Currently glium doesn't seem to have great support for:
    //   * resident textures (specifically, can't use them with samplers)
    //   * regular multidraw (though multidraw indirect may work)
    //
    // Try to incorporate both.

    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new()
        .with_depth_buffer(24)
        .build_glium().unwrap();

    let faces = [
        Face {
            position: [SIZE as f32, 0.0, 0.0].into(),
            normal: [1.0, 0.0, 0.0].into(),
        },
        Face {
            position: [0.0, SIZE as f32, 0.0].into(),
            normal: [0.0, 1.0, 0.0].into(),
        },
        Face {
            position: [0.0, 0.0, SIZE as f32].into(),
            normal: [0.0, 0.0, 1.0].into(),
        },
        Face {
            position: [-(SIZE as f32), 0.0, 0.0].into(),
            normal: [-1.0, 0.0, 0.0].into(),
        },
        Face {
            position: [0.0, -(SIZE as f32), 0.0].into(),
            normal: [0.0, -1.0, 0.0].into(),
        },
        Face {
            position: [0.0, 0.0, -(SIZE as f32)].into(),
            normal: [0.0, 0.0, -1.0].into(),
        },
    ];

    let mut mobs = [
        Mob {
            face: 3,
            position: [-(SIZE as f32), -(SIZE as f32), -(SIZE as f32)].into(),
            // position: [SIZE as f32/* - 1.0 / (2.0 * SIZE as f32)*/, SIZE as f32/* - 1.0 / (2.0 * SIZE as f32)*/, SIZE as f32/* - 1.0 / (2.0 * SIZE as f32)*/].into(),
            // velocity: [0.0, 0.0, 0.0].into(),
            velocity: Vector3::from([0.0, 1.0, 0.0]),
        },
        Mob {
            face: 3,
            position: [-(SIZE as f32), -(SIZE as f32), SIZE as f32].into(),
            // position: [SIZE as f32/* - 1.0 / (2.0 * SIZE as f32)*/, SIZE as f32/* - 1.0 / (2.0 * SIZE as f32)*/, SIZE as f32/* - 1.0 / (2.0 * SIZE as f32)*/].into(),
            // velocity: [0.0, 0.0, 0.0].into(),
            //velocity: [0.0, 1.0, 0.0].into(),
            velocity: Vector3::from([0.0, 1.0, -1.0]).normalize(),
        },
        Mob {
            face: 0,
            position: [SIZE as f32, 0.0, 0.0].into(),
            velocity: [0.0, 0.0, 1.0].into(),
        },
        Mob {
            face: 4,
            position: [0.0, -(SIZE as f32), -(SIZE as f32)].into(),
            velocity: [-2.0, 0.0, 0.0].into(),
        },
        Mob {
            face: 2,
            position: [SIZE as f32, SIZE as f32, SIZE as f32].into(),
            velocity: [0.0, -0.5, 0.0].into(),
        },
    ];
    let initial = [0.0, 0.0, 1.0].into();
    let face_shapes = faces.into_iter().map( |&Face { position, normal }| {
        fn orthogonal(v: Vector3<f32>) -> Vector3<f32>
        {
            let x = v.x.abs();
            let y = v.y.abs();
            let z = v.z.abs();

            let other = if x < y {
                if x < z { [1.0, 0.0, 0.0] }
                else { [0.0, 0.0, 1.0] }
            } else {
                if y < z { [0.0, 1.0, 0.0] }
                else { [0.0, 0.0, 1.0] }
            };
            v.cross(other.into())
        }
        let norm = normal.dot(initial);
        let q = if norm == -1.0 {
            Quaternion::from_sv(0.0, orthogonal(initial).normalize())
        } else {
            Quaternion::between_vectors(initial, normal)
        };
        let vertices = [
            FaceVertex { position: (q * Vector3::from([-(SIZE as f32), SIZE as f32, 0.0]) + position).into(), normal: normal.into(), tex_coords: [-(SIZE as f32), SIZE as f32], },
            FaceVertex { position: (q * Vector3::from([SIZE as f32, SIZE as f32, 0.0]) + position).into(), normal: normal.into(), tex_coords: [SIZE as f32, SIZE as f32], },
            FaceVertex { position: (q * Vector3::from([-(SIZE as f32), -(SIZE as f32), 0.0]) + position).into(), normal: normal.into(), tex_coords: [-(SIZE as f32), -(SIZE as f32)], },
            FaceVertex { position: (q * Vector3::from([SIZE as f32, -(SIZE as f32), 0.0]) + position).into(), normal: normal.into(), tex_coords: [SIZE as f32, -(SIZE as f32)], },
        ];
        println!("{:?}", &vertices);
        glium::vertex::VertexBuffer::new(&display, &vertices).unwrap()
    }).collect::<Vec<_>>();

    const GRID_WIDTH: usize = 128;
    const GRID_HEIGHT: usize = 128;
    const TILE_COUNT: usize = 4;

    let mut grid_textures = [[(0.0f32, 0.0f32, 0.0f32, 0.0f32); GRID_WIDTH * GRID_HEIGHT]; TILE_COUNT];
    for (k, grid_texture) in grid_textures.iter_mut().enumerate() {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                grid_texture[j*GRID_WIDTH + i] = if k & 1 == 1 && i < GRID_WIDTH / 16 || k & 2 == 2 && j < GRID_HEIGHT / 16 { (0.0, 0.0, 0.0, 1.0) } else { (0.0f32 as f32, 0.1f32, 0.2f32, 1.0f32) };
            }
        }
    }

    let grid_textures = grid_textures.into_iter().map( |grid_texture| glium::texture::RawImage2d {
        data: grid_texture.as_ref().into(),
        width: GRID_WIDTH as u32,
        height: GRID_HEIGHT as u32,
        format: glium::texture::ClientFormat::F32F32F32F32,
    } ).collect::<Vec<_>>();
    let grid_texture = glium::texture::Texture2dArray::new(&display, grid_textures).unwrap();

    let vertex1 = Vertex { position: [-0.5, -0.5, 0.0], normal: [0.0, 1.0, 1.0], };
    let vertex2 = Vertex { position: [ 0.0,  0.5, 0.0], normal: [1.0, 0.0, 1.0], };
    let vertex3 = Vertex { position: [ 0.5, -0.25, 0.0], normal: [0.0, 1.0, 1.0], };
    let shape = vec![vertex1, vertex2, vertex3];
    let positions = glium::VertexBuffer::new(&display, &shape).unwrap();
    /* let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[
        0u16, 1, 2
    ]).unwrap(); */
    // let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
    let program = glium::Program::from_source(&display, include_str!("shaders/mob_vertex.glsl"), include_str!("shaders/mob_fragment.glsl"), None).unwrap();
    let cube_program = glium::Program::from_source(&display, include_str!("shaders/cube_vertex.glsl"), include_str!("shaders/cube_fragment.glsl"), None).unwrap();

    // depth buffer test
    let cube_params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        // polygon_mode: glium::draw_parameters::PolygonMode::Line,
        backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
        .. Default::default()
    };

    // depth buffer test
    let params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        // backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
        .. Default::default()
    };

    let mut angle = 0f32;
    // let mut t: f32 = 0.1f32;
    let start = PreciseTime::now();
    let mut frames = 0;
    let mut cx = -20.0;
    let mut cy = 20.0;
    let mut cz = -60.0;
    'game: loop {
        // t = 0.1;
        angle += 0.005f32;

        // the direction of the light
        let light = [1.4, 0.4, -0.7f32];

        // the view matrix
        let view = view_matrix(&[cx, cy, cz], /*&[25.0, -15.0, 60.0]*/&[-cx, -cy, -cz], &[0.0, 1.0, 0.0]);
        //let view = view_matrix(&[5.0, -10.0, 15.0], &[0.1, 2.0, -2.0], &[0.0, 1.0, 0.0]);

        let mut target = display.draw();

        // perspective
        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;

            let f = 1.0 / (fov / 2.0).tan();

            [
                [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                [         0.0         ,     f ,              0.0              ,   0.0],
                [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
            ]
        };

        // clear
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        // faces
        for face in &face_shapes {
            // model matrix
            let model = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32],
            ];

            let grid_texture_sample = grid_texture
                .sampled()
                .wrap_function(glium::uniforms::SamplerWrapFunction::Repeat)
                .minify_filter(glium::uniforms::MinifySamplerFilter::LinearMipmapLinear)
                .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
                .anisotropy(16);

            // println!("{:?}", mob);
            // mob.physics(&faces, 0.05);
            // uniforms
            let uniforms = uniform! {
                u_light: light,
                model: model,
                view: view,
                perspective: perspective,
                diffuse_tex: grid_texture_sample,
            };

            /* let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[
                0u16, 1, 2,
                1, 2, 3,
            ]).unwrap();*/
            // draw the triangle here

            target.draw(face, /*&indices*/glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip), &cube_program, /*&glium::uniforms::EmptyUniforms*/&uniforms,
                &cube_params).unwrap();
        }

        let len = mobs.len() as f32;
        for (d, mob) in mobs.iter_mut().enumerate() {
            let d = f32::consts::PI * 2.0 / len * d as f32 + angle;
            // model matrix
            let model = [
                [d.cos(), 0.0, d.sin(), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-d.sin(), 0.0, d.cos(), 0.0],
                [mob.position.x, mob.position.y, mob.position.z, 1.0f32],
            ];

            // println!("{:?}", mob);
            mob.physics(&faces, 0.05);
            // uniforms
            let uniforms = uniform! {
                u_light: light,
                model: model,
                view: view,
                perspective: perspective,
            };

            // draw the triangle here
            target.draw(&positions, /*&indices*/glium::index::NoIndices(glium::index::PrimitiveType::/*TriangleStrip*/TrianglesList), &program, /*&glium::uniforms::EmptyUniforms*/&uniforms,
                &params).unwrap();
        }

        target.finish().unwrap();
        frames += 1;

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Left)) => {
                    cx -= 5.0;
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Right)) => {
                    cx += 5.0;
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Up)) => {
                    cz += 5.0;
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Down)) => {
                    cz -= 5.0;
                },
                glium::glutin::Event::Closed => break 'game,
                _ => ()
            }
        }
    }

    let end = PreciseTime::now();
    let duration = start.to(end).num_nanoseconds().map( |ns| frames as f64 * 1_000_000_000.0 / ns as f64 );
    println!("fps: {:?}", duration);
}
