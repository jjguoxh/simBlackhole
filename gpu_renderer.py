import os
import math
import struct
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    Image = None

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    GL_COMPUTE_SHADER,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glUseProgram,
    glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glBindImageTexture,
    GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST,
    glDispatchCompute, glMemoryBarrier, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
    glGetTexImage,
    glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
    GL_UNIFORM_BUFFER, GL_STATIC_DRAW
)
import glfw


def _compile_compute_shader(source: str) -> int:
    shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, source)
    glCompileShader(shader)
    status = glGetShaderiv(shader, 0x8B81)  # GL_COMPILE_STATUS
    if not status:
        log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Compute shader compile error:\n{log}")
    program = glCreateProgram()
    glAttachShader(program, shader)
    glLinkProgram(program)
    link_status = glGetProgramiv(program, 0x8B82)  # GL_LINK_STATUS
    if not link_status:
        log = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return program


def _create_context(width: int = 1, height: int = 1):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW for OpenGL context")
    # Request core 4.3 for compute shaders
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden window
    window = glfw.create_window(width, height, "GPU Renderer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window/context")
    glfw.make_context_current(window)
    return window


def _destroy_context(window):
    glfw.destroy_window(window)
    glfw.terminate()


def _ubo_camera(cam_pos: Tuple[float, float, float], cam_right, cam_up, cam_forward, tan_half_fov: float, aspect: float, moving: bool) -> bytes:
    # std140: vec3 padded to 16 bytes, floats are 4 bytes, bool as 4-byte int
    def pack_vec3(v):
        return struct.pack('4f', v[0], v[1], v[2], 0.0)
    data = bytearray()
    data += pack_vec3(cam_pos)
    data += pack_vec3(cam_right)
    data += pack_vec3(cam_up)
    data += pack_vec3(cam_forward)
    data += struct.pack('f', tan_half_fov)
    data += struct.pack('f', aspect)
    data += struct.pack('i', 1 if moving else 0)
    data += struct.pack('i', 0)  # _pad4
    return bytes(data)


def _ubo_disk(r1: float, r2: float) -> bytes:
    # disk_r1, disk_r2, disk_num, thickness
    return struct.pack('4f', r1, r2, 1.0, 0.0)


def _ubo_objects() -> bytes:
    # numObjects + arrays (empty for now)
    data = bytearray()
    data += struct.pack('i', 0)  # numObjects
    data += struct.pack('i', 0)  # std140 alignment for int -> 16 bytes? We'll add padding
    data += struct.pack('i', 0)
    data += struct.pack('i', 0)
    # objPosRadius[16] vec4
    data += struct.pack('64f', *([0.0] * 64))
    # objColor[16] vec4
    data += struct.pack('64f', *([0.0] * 64))
    # mass[16] float
    data += struct.pack('16f', *([0.0] * 16))
    return bytes(data)


def render_png_bytes(width: int = 200, height: int = 150, azimuth_deg: float = 0.0, elevation_deg: float = 90.0, radius: float = 6.34e10, fov_deg: float = 60.0, moving: bool = False, disk_r1: float = 2.0e10, disk_r2: float = 7.0e10) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow not installed: pip install pillow")

    # Compute camera vectors (match CPU-geodesic.cpp basis)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    target = (0.0, 0.0, 0.0)
    cam_pos = (
        target[0] + radius * math.sin(el) * math.cos(az),
        target[1] + radius * math.cos(el),
        target[2] + radius * math.sin(el) * math.sin(az),
    )
    fx, fy, fz = target[0] - cam_pos[0], target[1] - cam_pos[1], target[2] - cam_pos[2]
    fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    forward = (fx / fl, fy / fl, fz / fl)
    right = (forward[2], 0.0, -forward[0])
    rl = math.sqrt(right[0] ** 2 + right[1] ** 2 + right[2] ** 2) or 1.0
    right = (right[0] / rl, right[1] / rl, right[2] / rl)
    up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    tan_half_fov = math.tan(math.radians(fov_deg) * 0.5)
    aspect = width / float(height)

    # GL init
    window = _create_context()
    try:
        # Texture as image binding=0
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindImageTexture(0, tex, 0, False, 0, 0x88B9, GL_RGBA8)  # GL_WRITE_ONLY

        # UBO: Camera (binding=1)
        cam_buf = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, cam_buf)
        cam_bytes = _ubo_camera(cam_pos, right, up, forward, tan_half_fov, aspect, moving)
        glBufferData(GL_UNIFORM_BUFFER, len(cam_bytes), cam_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cam_buf)

        # UBO: Disk (binding=2)
        disk_buf = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, disk_buf)
        disk_bytes = _ubo_disk(disk_r1, disk_r2)
        glBufferData(GL_UNIFORM_BUFFER, len(disk_bytes), disk_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, disk_buf)

        # UBO: Objects (binding=3)
        obj_buf = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, obj_buf)
        obj_bytes = _ubo_objects()
        glBufferData(GL_UNIFORM_BUFFER, len(obj_bytes), obj_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, obj_buf)

        # Compile compute shader
        root = Path(__file__).resolve().parents[1]
        shader_path = root / 'geodesic.comp'
        with open(shader_path, 'r', encoding='utf-8') as f:
            src = f.read()
        program = _compile_compute_shader(src)
        glUseProgram(program)

        # Dispatch
        gx = (width + 15) // 16
        gy = (height + 15) // 16
        glDispatchCompute(gx, gy, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Readback
        buf = bytearray(width * height * 4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        # Convert RGBA to RGB by dropping alpha
        rgb = bytearray()
        for i in range(0, len(buf), 4):
            rgb.extend(buf[i:i+3])
        img = Image.frombytes('RGB', (width, height), bytes(rgb))
        import io
        out = io.BytesIO()
        img.save(out, format='PNG')
        return out.getvalue()
    finally:
        _destroy_context(window)