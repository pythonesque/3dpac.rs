extern crate cgmath;
#[macro_use]
extern crate glium;

use cgmath::{Angle, BaseFloat, BaseNum, EuclideanVector, One, Quaternion, Rad, Rotation, Rotation3, Point, Point2, Point3, Vector, Vector2, Vector3, Zero};

use std::f32;

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
        let l = self.position + self.velocity * t;
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
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new()
        .with_depth_buffer(24)
        .build_glium().unwrap();

    let faces = [
        Face {
            position: [SIZE as f32, 0.0, 0.0].into(),
            normal: [-1.0, 0.0, 0.0].into(),
        },
        Face {
            position: [0.0, SIZE as f32, 0.0].into(),
            normal: [0.0, -1.0, 0.0].into(),
        },
        Face {
            position: [0.0, 0.0, SIZE as f32].into(),
            normal: [0.0, 0.0, -1.0].into(),
        },
        Face {
            position: [-(SIZE as f32), 0.0, 0.0].into(),
            normal: [1.0, 0.0, 0.0].into(),
        },
        Face {
            position: [0.0, -(SIZE as f32), 0.0].into(),
            normal: [0.0, 1.0, 0.0].into(),
        },
        Face {
            position: [0.0, 0.0, -(SIZE as f32)].into(),
            normal: [0.0, 0.0, 1.0].into(),
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
    let face_shapes = faces.into_iter().map( |&Face { position, normal }| {
        // let basis = Quaternion::from_axis_angle(board[self.face].normal, board[nearest_face.0].normal)
        // let basis = Quaternion::look_at(position.into(), normal.into());
        let initial = [0.0, 0.0, -1.0].into();
        let q = if initial == -normal {
            Quaternion::between_vectors([0.0, 0.0, 1.0].into(), normal)
        } else {
            Quaternion::between_vectors(initial, normal)
        };
        // let position = q.rotate_vector(self.velocity);
        let vertices = [
            FaceVertex { position: (q * Vector3::from([-(SIZE as f32), SIZE as f32, 0.0]) + position).into(), normal: (-normal).into(), tex_coords: [-(SIZE as f32), SIZE as f32], },
            FaceVertex { position: (q * Vector3::from([SIZE as f32, SIZE as f32, 0.0]) + position).into(), normal: (-normal).into(), tex_coords: [SIZE as f32, SIZE as f32], },
            FaceVertex { position: (q * Vector3::from([-(SIZE as f32), -(SIZE as f32), 0.0]) + position).into(), normal: (-normal).into(), tex_coords: [-(SIZE as f32), -(SIZE as f32)], },
            FaceVertex { position: (q * Vector3::from([SIZE as f32, -(SIZE as f32), 0.0]) + position).into(), normal: (-normal).into(), tex_coords: [SIZE as f32, -(SIZE as f32)], },
        ];
        println!("{:?}", &vertices);
        glium::vertex::VertexBuffer::new(&display, &vertices).unwrap()
        // let basis = Quaternion::from_axis_angle(board[self.face].normal, Angle::turn_div_4());
        // let shape = glium::vertex::VertexBuffer::new(&display, &[
        //         Vertex { position: [-1.0,  1.0, 0.0], normal: [0.0, 0.0, -1.0] },
        //         Vertex { position: [ 1.0,  1.0, 0.0], normal: [0.0, 0.0, -1.0] },
        //         Vertex { position: [-1.0, -1.0, 0.0], normal: [0.0, 0.0, -1.0] },
        //         Vertex { position: [ 1.0, -1.0, 0.0], normal: [0.0, 0.0, -1.0] },
        //     ]).unwrap();
    }).collect::<Vec<_>>();


    let vertex1 = Vertex { position: [-0.5, -0.5, 0.0], normal: [0.0, 1.0, 1.0], };
    let vertex2 = Vertex { position: [ 0.0,  0.5, 0.0], normal: [1.0, 0.0, 1.0], };
    let vertex3 = Vertex { position: [ 0.5, -0.25, 0.0], normal: [0.0, 1.0, 1.0], };
    let shape = vec![vertex1, vertex2, vertex3];
    let positions = glium::VertexBuffer::new(&display, &shape).unwrap();
    /* let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[
        0u16, 1, 2
    ]).unwrap(); */
    // let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
    let vertex_shader_src = r#"
        #version 150

        in vec3 position;
        in vec3 normal;

        out vec3 v_normal;
        out vec3 v_position;

        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            mat4 modelview = view * model;
            v_normal = transpose(inverse(mat3(modelview))) * normal;
            // gl_Position = vec4(position, 0.0, 1.0);
            gl_Position = perspective * modelview * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;
    let fragment_shader_src = r#"
        #version 140

        in vec3 v_normal;
        in vec3 v_position;

        out vec4 color;

        uniform vec3 u_light;

        const vec3 ambient_color = vec3(0.2, 0.0, 0.0);
        const vec3 diffuse_color = vec3(0.6, 0.0, 0.0);
        const vec3 specular_color = vec3(1.0, 1.0, 1.0);

        void main() {
            float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);

            vec3 camera_dir = normalize(-v_position);
            vec3 half_direction = normalize(normalize(u_light) + camera_dir);
            float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);

            color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);

            // float brightness = dot(normalize(v_normal), normalize(u_light));
            // vec3 dark_color = vec3(0.6, 0.0, 0.0);
            // vec3 regular_color = vec3(1.0, 0.0, 0.0);
            // color = vec4(1.0, 0.0, 0.0, 1.0);
            // color = vec4(mix(dark_color, regular_color, brightness), 1.0);
        }
    "#;
    let cube_vertex_shader_src = r#"
        #version 150

        in vec3 position;
        in vec3 normal;
        in vec2 tex_coords;

        out vec3 v_normal;
        out vec3 v_position;
        out vec2 v_tex_coords;

        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            v_tex_coords = tex_coords;
            // if (
            mat4 modelview = view * model;
            v_normal = transpose(inverse(mat3(modelview))) * normal;
            // gl_Position = vec4(position, 0.0, 1.0);
            gl_Position = perspective * modelview * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;
    let cube_fragment_shader_src = r#"
        #version 140

        in vec3 v_normal;
        in vec3 v_position;
        in vec2 v_tex_coords;

        out vec4 color;

        uniform vec3 u_light;

        const vec3 ambient_color = vec3(0.0, 0.0, 0.1);
        const vec3 diffuse_color = vec3(0.2, 0.2, 0.2);
        const vec3 specular_color = vec3(0.0, 0.1, 0.0);
        const vec3 grid_ambient_color = vec3(0.0, 0.0, 0.0);
        const vec3 grid_diffuse_color = vec3(0.2, 0.0, 0.0);
        const vec3 grid_specular_color = vec3(0.2, 0.0, 0.0);

        mat3 cotangent_frame(vec3 normal, vec3 pos, vec2 uv) {
            vec3 dp1 = dFdx(pos);
            vec3 dp2 = dFdy(pos);
            vec2 duv1 = dFdx(uv);
            vec2 duv2 = dFdy(uv);
            vec3 dp2perp = cross(dp2, normal);
            vec3 dp1perp = cross(normal, dp1);
            vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
            vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
            float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
            return mat3(T * invmax, B * invmax, normal);
        }

        void main() {
            vec3 camera_dir = normalize(-v_position);
            vec3 half_direction = normalize(normalize(u_light) + camera_dir);

            /// mat3 tbn = cotangent_frame(v_position, v_normal, vec2(0, 0));
            if(fract(v_tex_coords.x) < 0.1f || fract(v_tex_coords.y) < 0.1f) {
                /*float dist = 0;
                if (fract(v_tex_coords.x) < 0.01f) {
                  dist += fract(v_tex_coords.x) * fract(v_tex_coords.x);
                }
                if (fract(v_tex_coords.y) < 0.01f) {
                  dist += fract(v_tex_coords.y) * fract(v_tex_coords.y);
                }
                dist = sqrt(dist);
                // Idea: based on distance to the grid line, we make the hole progressively deeper.
                float x_angle;
                if (fract(v_tex_coords.x) < 0.01f) {
                  // y axis, angle is with x and z.
                  // cos theta = fract(v_tex_coords.x) / dist.
                  x_angle = acos(fract(v_tex_coords.x) / dist);
                } else {
                  x_angle = 0.0;
                }
                float y_angle;
                if (fract(v_tex_coords.y) < 0.01f) {
                  // x axis, angle is with y and z.
                  // sin theta = fract(v_tex_coords.y) / dist;
                  y_angle = asin(fract(v_tex_coords.y) / 0.01f);
                } else {
                  y_angle = 0.0;
                }
                float normal_map;
                // Attempt to make a normal pointing inward at an angle proportional to distance to
                // the grid interior.
                if (fract(v_tex_coords.x) < fract(v_tex_coords.y)) {
                    float dist = sqrt(0.01f - fract(v_tex_coords.x) * fract(v_tex_coords.x));
                    normal_map = vec3(, 0.0, );
                } else {
                    // x axis, angle is with y and z.
                    float dist = sqrt(0.01f - fract(v_tex_coords.y) * fract(v_tex_coords.y));
                }
                vec3 normal_map = vec3(dist, );
                mat3 tbn = cotangent_frame(v_position, v_normal, v_tex_coords);
                vec3 real_normal = normalize(tbn * -(normal_map * 2.0 - 1.0));*/
                float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
                float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);
                color = vec4(grid_ambient_color + diffuse * grid_diffuse_color + specular * grid_specular_color, 1.0);
            } else {
                float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
                float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);

                color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
            }

            // float brightness = dot(normalize(v_normal), normalize(u_light));
            // vec3 dark_color = vec3(0.6, 0.0, 0.0);
            // vec3 regular_color = vec3(1.0, 0.0, 0.0);
            // color = vec4(1.0, 0.0, 0.0, 1.0);
            // color = vec4(mix(dark_color, regular_color, brightness), 1.0);
        }
    "#;
    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let cube_program = glium::Program::from_source(&display, cube_vertex_shader_src, cube_fragment_shader_src, None).unwrap();

    // depth buffer test
    let cube_params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        // polygon_mode: glium::draw_parameters::PolygonMode::Line,
        // backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,            
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
    loop {
        // t = 0.1;
        angle += 0.005f32;

        // the direction of the light
        let light = [1.4, 0.4, -0.7f32];

        // the view matrix
        let view = view_matrix(&[-20.0, 20.0, -60.0], &[25.0, -15.0, 60.0], &[0.0, 1.0, 0.0]);
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

            // println!("{:?}", mob);
            // mob.physics(&faces, 0.05);
            // uniforms
            let uniforms = uniform! {
                u_light: light,
                model: model,
                view: view,
                perspective: perspective,
            };

            let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[
                0u16, 1, 2,
                1, 2, 3,
            ]).unwrap();
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

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                _ => ()
            }
        }
    }
}
