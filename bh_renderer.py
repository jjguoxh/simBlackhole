import math
from dataclasses import dataclass
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # Will be checked at runtime


# Physical constants
c = 299_792_458.0
G = 6.67430e-11


@dataclass
class BlackHole:
    mass: float  # kg

    @property
    def r_s(self) -> float:
        return 2.0 * G * self.mass / (c * c)


@dataclass
class Camera3D:
    azimuth: float  # radians
    elevation: float  # radians
    radius: float
    fov_deg: float
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def update_pos(self) -> Tuple[float, float, float]:
        tx, ty, tz = self.target
        x = tx + self.radius * math.sin(self.elevation) * math.cos(self.azimuth)
        y = ty + self.radius * math.cos(self.elevation)
        z = tz + self.radius * math.sin(self.elevation) * math.sin(self.azimuth)
        return (x, y, z)

    @property
    def tan_half_fov(self) -> float:
        return math.tan(math.radians(self.fov_deg) * 0.5)

    def basis(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        # forward = normalize(target - pos)
        px, py, pz = self.update_pos()
        tx, ty, tz = self.target
        fx, fy, fz = tx - px, ty - py, tz - pz
        fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / fl, fy / fl, fz / fl
        # right = normalize(cross(forward, upWorld)) with upWorld=(0,1,0)
        rx = fz
        ry = 0.0
        rz = -fx
        rl = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
        rx, ry, rz = rx / rl, ry / rl, rz / rl
        # up = cross(right, forward)
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        return (fx, fy, fz), (rx, ry, rz), (ux, uy, uz)


@dataclass
class Object3D:
    center: Tuple[float, float, float]
    radius: float
    color: Tuple[float, float, float, float]  # RGBA


@dataclass
class Disk:
    r1: float
    r2: float


@dataclass
class Ray3D:
    r: float
    theta: float
    phi: float
    dr: float
    dtheta: float
    dphi: float
    E: float


def init_ray_state(pos: Tuple[float, float, float], dir3: Tuple[float, float, float], rs: float) -> Ray3D:
    x, y, z = pos
    dx, dy, dz = dir3
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-9:
        r = 1e-9
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    # spherical basis
    er = (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )
    e_theta = (
        math.cos(theta) * math.cos(phi),
        math.cos(theta) * math.sin(phi),
        -math.sin(theta),
    )
    e_phi = (-math.sin(phi), math.cos(phi), 0.0)

    dr = dx * er[0] + dy * er[1] + dz * er[2]
    dtheta = (dx * e_theta[0] + dy * e_theta[1] + dz * e_theta[2]) / r
    denom = r * max(math.sin(theta), 1e-9)
    dphi = (dx * e_phi[0] + dy * e_phi[1] + dz * e_phi[2]) / denom

    f = 1.0 - rs / r
    dt_dlambda = math.sqrt(max((dr * dr) / max(f, 1e-12) + r * r * (dtheta * dtheta + (math.sin(theta) ** 2) * dphi * dphi), 0.0))
    E = f * dt_dlambda
    return Ray3D(r=r, theta=theta, phi=phi, dr=dr, dtheta=dtheta, dphi=dphi, E=E)


def geodesic_rhs_3d(ray: Ray3D, rs: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    r = max(ray.r, 1e-9)
    theta = ray.theta
    dr = ray.dr
    dtheta = ray.dtheta
    dphi = ray.dphi
    f = 1.0 - rs / r
    dt_dlambda = ray.E / max(f, 1e-12)

    d1 = (dr, dtheta, dphi)
    d2_r = - (rs / (2.0 * r * r)) * f * dt_dlambda * dt_dlambda + (rs / (2.0 * r * r * max(f, 1e-12))) * dr * dr + r * (dtheta * dtheta + (math.sin(theta) ** 2) * dphi * dphi)
    d2_theta = - (2.0 / r) * dr * dtheta + math.sin(theta) * math.cos(theta) * dphi * dphi
    d2_phi = - (2.0 / r) * dr * dphi - 2.0 * math.cos(theta) / max(math.sin(theta), 1e-9) * dtheta * dphi
    d2 = (d2_r, d2_theta, d2_phi)
    return d1, d2


def rk4_step_3d(ray: Ray3D, d_lambda: float, rs: float) -> Ray3D:
    k1a, k1b = geodesic_rhs_3d(ray, rs)
    r2 = Ray3D(
        r=ray.r + d_lambda * 0.5 * k1a[0],
        theta=ray.theta + d_lambda * 0.5 * k1a[1],
        phi=ray.phi + d_lambda * 0.5 * k1a[2],
        dr=ray.dr + d_lambda * 0.5 * k1b[0],
        dtheta=ray.dtheta + d_lambda * 0.5 * k1b[1],
        dphi=ray.dphi + d_lambda * 0.5 * k1b[2],
        E=ray.E,
    )
    k2a, k2b = geodesic_rhs_3d(r2, rs)
    r3 = Ray3D(
        r=ray.r + d_lambda * 0.5 * k2a[0],
        theta=ray.theta + d_lambda * 0.5 * k2a[1],
        phi=ray.phi + d_lambda * 0.5 * k2a[2],
        dr=ray.dr + d_lambda * 0.5 * k2b[0],
        dtheta=ray.dtheta + d_lambda * 0.5 * k2b[1],
        dphi=ray.dphi + d_lambda * 0.5 * k2b[2],
        E=ray.E,
    )
    k3a, k3b = geodesic_rhs_3d(r3, rs)
    r4 = Ray3D(
        r=ray.r + d_lambda * k3a[0],
        theta=ray.theta + d_lambda * k3a[1],
        phi=ray.phi + d_lambda * k3a[2],
        dr=ray.dr + d_lambda * k3b[0],
        dtheta=ray.dtheta + d_lambda * k3b[1],
        dphi=ray.dphi + d_lambda * k3b[2],
        E=ray.E,
    )
    k4a, k4b = geodesic_rhs_3d(r4, rs)

    r = ray.r + (d_lambda / 6.0) * (k1a[0] + 2 * k2a[0] + 2 * k3a[0] + k4a[0])
    theta = ray.theta + (d_lambda / 6.0) * (k1a[1] + 2 * k2a[1] + 2 * k3a[1] + k4a[1])
    phi = ray.phi + (d_lambda / 6.0) * (k1a[2] + 2 * k2a[2] + 2 * k3a[2] + k4a[2])
    dr = ray.dr + (d_lambda / 6.0) * (k1b[0] + 2 * k2b[0] + 2 * k3b[0] + k4b[0])
    dtheta = ray.dtheta + (d_lambda / 6.0) * (k1b[1] + 2 * k2b[1] + 2 * k3b[1] + k4b[1])
    dphi = ray.dphi + (d_lambda / 6.0) * (k1b[2] + 2 * k2b[2] + 2 * k3b[2] + k4b[2])

    return Ray3D(r=r, theta=theta, phi=phi, dr=dr, dtheta=dtheta, dphi=dphi, E=ray.E)


def sph_to_cart(ray: Ray3D) -> Tuple[float, float, float]:
    x = ray.r * math.sin(ray.theta) * math.cos(ray.phi)
    y = ray.r * math.sin(ray.theta) * math.sin(ray.phi)
    z = ray.r * math.cos(ray.theta)
    return x, y, z


def intercept_horizon(ray: Ray3D, rs: float) -> bool:
    return ray.r <= rs


def crosses_equatorial(prev_pos: Tuple[float, float, float], new_pos: Tuple[float, float, float], disk: Disk) -> bool:
    crossed = prev_pos[1] * new_pos[1] < 0.0
    r_xz = math.sqrt(new_pos[0] * new_pos[0] + new_pos[2] * new_pos[2])
    return crossed and (r_xz >= disk.r1 and r_xz <= disk.r2)


def intercept_objects(pos: Tuple[float, float, float], objects: List[Object3D]) -> Tuple[bool, Object3D]:
    for o in objects:
        dx = pos[0] - o.center[0]
        dy = pos[1] - o.center[1]
        dz = pos[2] - o.center[2]
        if math.sqrt(dx * dx + dy * dy + dz * dz) <= o.radius:
            return True, o
    return False, None


def shade_object(pos: Tuple[float, float, float], cam_pos: Tuple[float, float, float], obj: Object3D) -> Tuple[int, int, int]:
    nx = pos[0] - obj.center[0]
    ny = pos[1] - obj.center[1]
    nz = pos[2] - obj.center[2]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    vx = cam_pos[0] - pos[0]
    vy = cam_pos[1] - pos[1]
    vz = cam_pos[2] - pos[2]
    vl = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    vx, vy, vz = vx / vl, vy / vl, vz / vl
    diff = max(nx * vx + ny * vy + nz * vz, 0.0)
    ambient = 0.1
    intensity = ambient + (1.0 - ambient) * diff
    r = int(255 * min(max(obj.color[0] * intensity, 0.0), 1.0))
    g = int(255 * min(max(obj.color[1] * intensity, 0.0), 1.0))
    b = int(255 * min(max(obj.color[2] * intensity, 0.0), 1.0))
    return r, g, b


class Renderer3D:
    def __init__(self, bh_mass: float, azimuth_deg: float, elevation_deg: float, radius: float, fov_deg: float, disk_r1: float, disk_r2: float, objects: List[Object3D] | None = None):
        self.bh = BlackHole(mass=bh_mass)
        self.camera = Camera3D(
            azimuth=math.radians(azimuth_deg),
            elevation=math.radians(elevation_deg),
            radius=radius,
            fov_deg=fov_deg,
        )
        self.disk = Disk(r1=disk_r1, r2=disk_r2)
        self.objects = objects or []
        # CPU 演示参数：较大的步长 + 较少的步数以提升速度
        self.d_lambda = 2e8
        self.max_steps = 4000
        self.escape_r = 1e14

    def render(self, width: int, height: int) -> Image:
        if Image is None:
            raise RuntimeError("Pillow (PIL) not installed. Please install with `pip install pillow`.")

        img = Image.new("RGB", (width, height))
        px = img.load()

        cam_pos = self.camera.update_pos()
        forward, right, up = self.camera.basis()
        aspect = width / float(height)
        thf = self.camera.tan_half_fov

        for y in range(height):
            v = (1.0 - 2.0 * ((y + 0.5) / float(height))) * thf
            for x in range(width):
                u = (2.0 * ((x + 0.5) / float(width)) - 1.0) * aspect * thf
                dirx = u * right[0] + v * up[0] + forward[0]
                diry = u * right[1] + v * up[1] + forward[1]
                dirz = u * right[2] + v * up[2] + forward[2]
                dl = math.sqrt(dirx * dirx + diry * diry + dirz * dirz) or 1.0
                dirx, diry, dirz = dirx / dl, diry / dl, dirz / dl

                ray = init_ray_state(cam_pos, (dirx, diry, dirz), self.bh.r_s)
                prev_pos = sph_to_cart(ray)

                hit_bh = False
                hit_disk = False
                hit_obj = None

                for _ in range(self.max_steps):
                    if intercept_horizon(ray, self.bh.r_s):
                        hit_bh = True
                        break
                    ray = rk4_step_3d(ray, self.d_lambda, self.bh.r_s)
                    pos = sph_to_cart(ray)
                    if crosses_equatorial(prev_pos, pos, self.disk):
                        hit_disk = True
                        break
                    hit, obj = intercept_objects(pos, self.objects)
                    if hit:
                        hit_obj = obj
                        break
                    prev_pos = pos
                    if ray.r > self.escape_r:
                        break

                if hit_bh:
                    px[x, y] = (0, 0, 0)
                elif hit_disk:
                    # 使用实际穿越盘面的当前位置计算半径比例，避免统一饱和色块
                    r_frac = min(max(math.sqrt(pos[0] ** 2 + pos[2] ** 2) / self.disk.r2, 0.0), 1.0)
                    px[x, y] = (int(255 * 1.0), int(255 * r_frac), int(255 * 0.2))
                elif hit_obj is not None:
                    # 对物体命中使用当前位置着色
                    px[x, y] = shade_object(pos, cam_pos, hit_obj)
                else:
                    px[x, y] = (0, 0, 0)

        return img


def render_png_bytes(
    width: int = 200,
    height: int = 150,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 90.0,
    radius: float = 6.34e10,
    fov_deg: float = 60.0,
    bh_mass: float = 8.2e36,
    disk_r1: float = 3.0e10,
    disk_r2: float = 3.8e10,
) -> bytes:
    # Example objects similar to C++ test scene
    objects = [
        Object3D(center=(0.0, 0.0, -5.0e10), radius=1.5e10, color=(1.0, 0.2, 0.2, 1.0)),
        Object3D(center=(3.0e10, 0.0, -7.0e10), radius=1.2e10, color=(0.2, 1.0, 0.2, 1.0)),
    ]

    renderer = Renderer3D(
        bh_mass=bh_mass,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        radius=radius,
        fov_deg=fov_deg,
        disk_r1=disk_r1,
        disk_r2=disk_r2,
        objects=objects,
    )
    img = renderer.render(width, height)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()