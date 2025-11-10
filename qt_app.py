import math
import struct
from pathlib import Path

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
    from PyQt5.QtGui import QSurfaceFormat
    from PyQt5.QtCore import Qt
except ImportError:
    # Fallback to PySide6 if PyQt5 is unavailable
    from PySide6.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtCore import Qt

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    GL_COMPUTE_SHADER, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glUseProgram,
    glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glBindImageTexture,
    GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST,
    glDispatchCompute, glMemoryBarrier, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
    glActiveTexture, GL_TEXTURE0, glGetUniformLocation, glUniform1i,
    glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
    GL_UNIFORM_BUFFER, GL_STATIC_DRAW,
    glGenVertexArrays, glBindVertexArray,
    glBindBuffer, GL_ARRAY_BUFFER, glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
    GL_FLOAT, glDrawArrays, GL_TRIANGLES,
    glViewport
)


def _compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    status = glGetShaderiv(shader, 0x8B81)  # GL_COMPILE_STATUS
    if not status:
        log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader


def _link_program(shaders: list[int]) -> int:
    program = glCreateProgram()
    for s in shaders:
        glAttachShader(program, s)
    glLinkProgram(program)
    link_status = glGetProgramiv(program, 0x8B82)  # GL_LINK_STATUS
    if not link_status:
        log = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return program


def _ubo_camera(cam_pos, cam_right, cam_up, cam_forward, tan_half_fov: float, aspect: float, moving: bool) -> bytes:
    # std140: vec3 padded to 16 bytes
    def pack_vec3(v):
        return struct.pack('4f', float(v[0]), float(v[1]), float(v[2]), 0.0)
    data = bytearray()
    data += pack_vec3(cam_pos)
    data += pack_vec3(cam_right)
    data += pack_vec3(cam_up)
    data += pack_vec3(cam_forward)
    data += struct.pack('f', float(tan_half_fov))
    data += struct.pack('f', float(aspect))
    data += struct.pack('i', 1 if moving else 0)
    data += struct.pack('i', 0)  # _pad4
    return bytes(data)


def _ubo_disk(r1: float, r2: float) -> bytes:
    return struct.pack('4f', float(r1), float(r2), 1.0, 0.0)


def _ubo_objects() -> bytes:
    data = bytearray()
    data += struct.pack('i', 0)  # numObjects
    # Arrays (unused when numObjects=0)
    data += struct.pack('64f', *([0.0] * 64))  # objPosRadius[16]
    data += struct.pack('64f', *([0.0] * 64))  # objColor[16]
    data += struct.pack('16f', *([0.0] * 16))  # mass[16]
    return bytes(data)


class GLBlackHoleWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        # Camera params
        self.azimuth_deg = 0.0
        self.elevation_deg = 90.0
        self.radius = 6.34e10
        self.fov_deg = 60.0
        self.moving = False
        # Disk params
        self.disk_r1 = 3.0e10
        self.disk_r2 = 3.8e10
        # Interaction
        self._last_pos = None

        # GL handles
        self.compute_program = None
        self.blit_program = None
        self.out_tex = None
        self.cam_ubo = None
        self.disk_ubo = None
        self.obj_ubo = None
        self.quad_vao = None
        self.quad_vbo = None

    def initializeGL(self):
        # Compile compute shader
        root = Path(__file__).resolve().parents[1]
        shader_path = root / 'geodesic.comp'
        with open(shader_path, 'r', encoding='utf-8') as f:
            comp_src = f.read()
        comp = _compile_shader(comp_src, GL_COMPUTE_SHADER)
        self.compute_program = _link_program([comp])

        # Fullscreen quad shaders
        vs_src = """
        #version 430
        layout(location=0) in vec2 aPos;
        out vec2 vUV;
        void main(){ vUV = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }
        """
        fs_src = """
        #version 430
        in vec2 vUV;
        uniform sampler2D uTex;
        out vec4 fragColor;
        void main(){ fragColor = texture(uTex, vUV); }
        """
        vs = _compile_shader(vs_src, GL_VERTEX_SHADER)
        fs = _compile_shader(fs_src, GL_FRAGMENT_SHADER)
        self.blit_program = _link_program([vs, fs])

        # Output texture (initial size = widget size)
        w, h = self.width(), self.height()
        self.out_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.out_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindImageTexture(0, self.out_tex, 0, False, 0, 0x88B9, GL_RGBA8)  # GL_WRITE_ONLY

        # UBOs
        self.cam_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.cam_ubo)
        cam_bytes = _ubo_camera((0,0,0), (1,0,0), (0,1,0), (0,0,1), math.tan(math.radians(self.fov_deg)*0.5), w/float(h), False)
        glBufferData(GL_UNIFORM_BUFFER, len(cam_bytes), cam_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.cam_ubo)

        self.disk_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        disk_bytes = _ubo_disk(self.disk_r1, self.disk_r2)
        glBufferData(GL_UNIFORM_BUFFER, len(disk_bytes), disk_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.disk_ubo)

        self.obj_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.obj_ubo)
        obj_bytes = _ubo_objects()
        glBufferData(GL_UNIFORM_BUFFER, len(obj_bytes), obj_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, self.obj_ubo)

        # Fullscreen quad geometry
        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)
        self.quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        import array
        verts = array.array('f', [
            -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,
        ])
        glBufferData(GL_ARRAY_BUFFER, verts.buffer_info()[1] * verts.itemsize, verts.tobytes(), GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, None)

        # Texture sampler binding
        glUseProgram(self.blit_program)
        loc = glGetUniformLocation(self.blit_program, b"uTex")
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(loc, 0)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        # Resize output texture
        glBindTexture(GL_TEXTURE_2D, self.out_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindImageTexture(0, self.out_tex, 0, False, 0, 0x88B9, GL_RGBA8)

    def paintGL(self):
        # Update camera UBO
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        target = (0.0, 0.0, 0.0)
        cam_pos = (
            target[0] + self.radius * math.sin(el) * math.cos(az),
            target[1] + self.radius * math.cos(el),
            target[2] + self.radius * math.sin(el) * math.sin(az),
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
        tan_half_fov = math.tan(math.radians(self.fov_deg) * 0.5)
        aspect = self.width() / float(max(self.height(), 1))

        glBindBuffer(GL_UNIFORM_BUFFER, self.cam_ubo)
        cam_bytes = _ubo_camera(cam_pos, right, up, forward, tan_half_fov, aspect, self.moving)
        glBufferData(GL_UNIFORM_BUFFER, len(cam_bytes), cam_bytes, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.cam_ubo)

        # Dispatch compute
        glUseProgram(self.compute_program)
        gx = (self.width() + 15) // 16
        gy = (self.height() + 15) // 16
        glDispatchCompute(max(gx, 1), max(gy, 1), 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Blit to screen
        glUseProgram(self.blit_program)
        glBindVertexArray(self.quad_vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.out_tex)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    # Interaction: drag to rotate, wheel to zoom
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self._last_pos is not None:
            x, y = event.x(), event.y()
            lx, ly = self._last_pos
            dx, dy = x - lx, y - ly
            self.azimuth_deg += dx * 0.4
            self.elevation_deg -= dy * 0.4
            self.elevation_deg = max(1.0, min(self.elevation_deg, 179.0))
            self._last_pos = (x, y)
            self.moving = True
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = None
            self.moving = False

    def wheelEvent(self, event):
        # Zoom by changing radius
        delta = event.angleDelta().y() / 120.0
        factor = math.pow(1.1, delta)
        self.radius = max(1.5e10, min(self.radius * factor, 1e12))
        self.update()


def main():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication([])
    win = QMainWindow()
    win.setWindowTitle("PyQt 黑洞光线追迹（GPU 实时）")
    widget = GLBlackHoleWidget()
    win.setCentralWidget(widget)
    win.resize(800, 600)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()