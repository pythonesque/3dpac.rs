#version 140

in vec3 v_normal;
in vec3 v_position;
in vec2 v_tex_coords;

out vec4 color;

uniform vec3 u_light;
uniform sampler2D diffuse_tex;

// const vec3 ambient_color = vec3(0.0, 0.0, 0.1);
// const vec3 diffuse_color = vec3(0.2, 0.2, 0.2);

const vec3 specular_color = vec3(0.5, 0.5, 0.5);
// const vec3 grid_ambient_color = vec3(0.0, 0.0, 0.0);
// const vec3 grid_diffuse_color = vec3(0.0, 0.0, 0.0);
// const vec3 grid_specular_color = vec3(0.0, 0.0, 0.0);

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

  vec3 diffuse_color = texture(diffuse_tex, v_tex_coords).rgb;
  vec3 ambient_color = diffuse_color * 0.2;
  // vec3 ambient_color = vec3(0.0, 0.0, diffuse_color.z * 0.5);

  /// mat3 tbn = cotangent_frame(v_position, v_normal, vec2(0, 0));
  //if(fract(v_tex_coords.x) < 0.1f || fract(v_tex_coords.y) < 0.1f) {
  //  /*float dist = 0;
  //  if (fract(v_tex_coords.x) < 0.01f) {
  //    dist += fract(v_tex_coords.x) * fract(v_tex_coords.x);
  //  }
  //  if (fract(v_tex_coords.y) < 0.01f) {
  //    dist += fract(v_tex_coords.y) * fract(v_tex_coords.y);
  //  }
  //  dist = sqrt(dist);
  //  // Idea: based on distance to the grid line, we make the hole progressively deeper.
  //  float x_angle;
  //  if (fract(v_tex_coords.x) < 0.01f) {
  //    // y axis, angle is with x and z.
  //    // cos theta = fract(v_tex_coords.x) / dist.
  //    x_angle = acos(fract(v_tex_coords.x) / dist);
  //  } else {
  //    x_angle = 0.0;
  //  }
  //  float y_angle;
  //  if (fract(v_tex_coords.y) < 0.01f) {
  //    // x axis, angle is with y and z.
  //    // sin theta = fract(v_tex_coords.y) / dist;
  //    y_angle = asin(fract(v_tex_coords.y) / 0.01f);
  //  } else {
  //    y_angle = 0.0;
  //  }
  //  float normal_map;
  //  // Attempt to make a normal pointing inward at an angle proportional to distance to
  //  // the grid interior.
  //  if (fract(v_tex_coords.x) < fract(v_tex_coords.y)) {
  //      float dist = sqrt(0.01f - fract(v_tex_coords.x) * fract(v_tex_coords.x));
  //      normal_map = vec3(, 0.0, );
  //  } else {
  //      // x axis, angle is with y and z.
  //      float dist = sqrt(0.01f - fract(v_tex_coords.y) * fract(v_tex_coords.y));
  //  }
  //  vec3 normal_map = vec3(dist, );
  //  mat3 tbn = cotangent_frame(v_position, v_normal, v_tex_coords);
  //  vec3 real_normal = normalize(tbn * -(normal_map * 2.0 - 1.0));*/
  //  /*float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
  //  float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);
  //  color = vec4(grid_ambient_color + diffuse * grid_diffuse_color + specular * grid_specular_color, 1.0);*/
  //  color = vec4(0);
  //} else {
  //  float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
  //  float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);

  //  color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
  //}
  float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
  float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);
  color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);

  // float brightness = dot(normalize(v_normal), normalize(u_light));
  // vec3 dark_color = vec3(0.6, 0.0, 0.0);
  // vec3 regular_color = vec3(1.0, 0.0, 0.0);
  // color = vec4(1.0, 0.0, 0.0, 1.0);
  // color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}
