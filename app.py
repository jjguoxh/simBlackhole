from flask import Flask, Response, request

try:
    from .bh_renderer import render_png_bytes as render_cpu
    from .gpu_renderer import render_png_bytes as render_gpu
except ImportError:
    from bh_renderer import render_png_bytes as render_cpu
    from gpu_renderer import render_png_bytes as render_gpu


app = Flask(__name__)


INDEX_HTML = """
<!doctype html>
<html lang=\"zh\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Python 黑洞光线追迹（3D）</title>
    <style>
      body { font-family: system-ui, Arial, sans-serif; margin: 20px; }
      img { border: 1px solid #ccc; display:block; }
      .row { margin-bottom: 10px; }
      label { margin-right: 10px; }
      input { width: 100px; }
    </style>
  </head>
  <body>
    <h2>Python 黑洞光线追迹（3D 近似，CPU快速版）</h2>
    <div class=\"row\">
      <label>宽度：<input id=\"w\" type=\"number\" value=\"200\" min=\"50\" max=\"400\"></label>
      <label>高度：<input id=\"h\" type=\"number\" value=\"150\" min=\"50\" max=\"300\"></label>
    </div>
    <div class=\"row\">
      <label>方位角(°)：<input id=\"az\" type=\"number\" value=\"0\"></label>
      <label>仰角(°)：<input id=\"el\" type=\"number\" value=\"90\"></label>
      <label>半径(m)：<input id=\"radius\" type=\"number\" value=\"6.34e10\"></label>
      <label>FOV(°)：<input id=\"fov\" type=\"number\" value=\"60\"></label>
    </div>
    <div class=\"row\">
      <label>盘内半径(m)：<input id=\"r1\" type=\"number\" value=\"3.0e10\"></label>
      <label>盘外半径(m)：<input id=\"r2\" type=\"number\" value=\"3.8e10\"></label>
    </div>
    <div class=\"row\">
      <label>模式：
        <select id=\"mode\">
          <option value=\"gpu\" selected>GPU (OpenGL)</option>
          <option value=\"cpu\">CPU</option>
        </select>
      </label>
      <button onclick=\"reloadImg()\">渲染</button>
    </div>
    <img id=\"img\" src=\"/render?w=200&h=150&mode=gpu&r1=3.0e10&r2=3.8e10\" alt=\"black hole render\" />
    <script>
      function reloadImg() {
        const w = document.getElementById('w').value;
        const h = document.getElementById('h').value;
        const az = document.getElementById('az').value;
        const el = document.getElementById('el').value;
        const radius = document.getElementById('radius').value;
        const fov = document.getElementById('fov').value;
        const r1 = document.getElementById('r1').value;
        const r2 = document.getElementById('r2').value;
        const mode = document.getElementById('mode').value;
        const url = `/render?w=${w}&h=${h}&az=${az}&el=${el}&radius=${radius}&fov=${fov}&r1=${r1}&r2=${r2}&mode=${mode}&t=${Date.now()}`;
        document.getElementById('img').src = url;
      }
    </script>
  </body>
</html>
"""


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/render")
def render():
    try:
        w = int(float(request.args.get("w", 200)))
        h = int(float(request.args.get("h", 150)))
        # 限制像素总量，避免CPU阻塞
        w = max(50, min(w, 400))
        h = max(50, min(h, 300))

        az = float(request.args.get("az", 0.0))
        el = float(request.args.get("el", 90.0))
        radius = float(request.args.get("radius", 6.34e10))
        fov = float(request.args.get("fov", 60.0))
        r1 = float(request.args.get("r1", 3.0e10))
        r2 = float(request.args.get("r2", 3.8e10))

        mode = request.args.get("mode", "gpu")
        if mode == "gpu":
            png = render_gpu(width=w, height=h, azimuth_deg=az, elevation_deg=el, radius=radius, fov_deg=fov, disk_r1=r1, disk_r2=r2)
        else:
            png = render_cpu(width=w, height=h, azimuth_deg=az, elevation_deg=el, radius=radius, fov_deg=fov, disk_r1=r1, disk_r2=r2)
        return Response(png, mimetype="image/png")
    except Exception as e:
        return Response(f"Error: {e}", status=500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)