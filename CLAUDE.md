写一个python + html demo，实现从一张上传图，计算 edge map，然后用 SDF/ PDE放大（如果有多个方法，在 html 上可以进行 side by side 对比）

---
下面是一份可以直接交给 **gemini-cli / codex** 去实现的「Python + HTML Demo」规格说明（Spec）。目标是：**上传一张图 → 计算 edge map → 用多种放大方法（至少 SDF 与 PDE）生成高分辨率线稿/边缘图 → HTML 里 side-by-side 对比**。包含明确的文件结构、接口、参数、算法细节（含公式与离散化方法）。

---

## 1. 目标与范围

### 目标（Must）

1. 用户在网页上传一张图片（JPG/PNG）。
2. 后端计算 **edge map**（Canny 为主）。
3. 将 edge map 进行放大（scale=2/4/8 可选），并实现至少两种方法：

   * **SDF 放大（Signed Distance Field / Distance Transform 驱动的重建）**
   * **PDE 放大（在 SDF 上做 PDE 迭代：reinitialization + curvature flow，或同级别 PDE 方法）**
4. 前端 HTML 以 **side-by-side**（同尺寸）展示：

   * 原图（可选）
   * 原分辨率 edge map
   * 放大结果：Baseline（bicubic/nearest） vs SDF vs PDE（至少两列）
5. 运行方式简单：`pip install -r requirements.txt` + `uvicorn ...`，本地打开页面可用。

### 非目标（Non-goals）

* 不做深度学习超分（不引入 torch）。
* 不追求完美的向量化（不做 svg 输出）。
* 不做复杂的全局状态与用户系统。

---

## 2. 技术选型

### 后端（Python）

* FastAPI + Uvicorn（轻量，便于 multipart 上传与静态文件）
* opencv-python（读图、Canny、resize、形态学）
* numpy（数值）
* scipy（`distance_transform_edt` 用于距离变换；若不想依赖 scipy，也可用 OpenCV 的 distanceTransform，但 scipy 更直观）

### 前端（HTML/JS/CSS）

* 原生 HTML + fetch（无框架）
* CSS flex/grid 做多列对比

---

## 3. 项目结构（必须按此组织，便于 codex 生成）

```
edge_sdf_pde_demo/
  README.md
  requirements.txt

  app/
    main.py
    imaging.py        # 读图、预处理、edge map
    sdf.py            # SDF 构建与 SDF upsample 渲染
    pde.py            # PDE 迭代（reinit + curvature flow）
    io_utils.py       # 保存 png、生成 id、路径

  static/
    index.html
    app.js
    style.css
    out/              # 输出图片（运行时生成/覆盖）
```

---

## 4. HTTP 接口设计

### 4.1 静态页面

* `GET /` → 返回 `static/index.html`
* `GET /static/...` → 静态文件（包含 `out/` 结果图）

### 4.2 图像处理 API

* `POST /api/process`
* Content-Type: `multipart/form-data`

#### 入参（form fields）

* `file`: 上传图片（必填）
* `scale`: int，默认 4，可选 {2, 4, 8}
* `canny_sigma`: float，默认 0.33（用于自动阈值）
* `edge_dilate`: int，默认 1（把细 edge 变成“有厚度的 stroke mask”，便于定义 signed distance）
* `aa_width`: float，默认 1.0（抗锯齿宽度，单位：高分辨率像素）
* `methods`: string，逗号分隔，例如 `"baseline,sdf,pde"`

#### 出参（JSON）

```json
{
  "id": "uuid",
  "input_url": "/static/out/uuid_input.png",
  "edge_url": "/static/out/uuid_edge.png",
  "results": {
    "baseline": "/static/out/uuid_baseline_x4.png",
    "sdf": "/static/out/uuid_sdf_x4.png",
    "pde": "/static/out/uuid_pde_x4.png"
  },
  "meta": {
    "scale": 4,
    "input_size": [H, W],
    "output_size": [H4, W4]
  }
}
```

---

## 5. 核心算法规范

> 重要约定：**我们“放大”的对象是 edge/stroke 图，而不是原彩图。**
> 流程：原图 → edge map（二值/灰度）→ stroke mask → SDF → upsample → 渲染输出。

### 5.1 Edge Map 计算（Canny，带自动阈值）

1. 读入 BGR/RGB，转灰度：

   * `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
2. 可选降噪（建议开）：

   * `gray_blur = cv2.GaussianBlur(gray, (5,5), 0)`
3. Canny 阈值自动估计（用 median）：

   * `v = median(gray_blur)`
   * `lower = max(0, (1 - sigma) * v)`
   * `upper = min(255, (1 + sigma) * v)`
4. Canny：

   * `edges = cv2.Canny(gray_blur, lower, upper, L2gradient=True)`
5. 二值化 edge map：

   * `E = edges > 0`（bool）

输出：

* `edge.png`：建议保存为 0/255 灰度图以便前端显示。

### 5.2 从 Edge 生成 Stroke Mask（用于 signed distance）

Canny 输出是“细线”，严格意义上没有 inside/outside，无法天然定义 signed distance。
因此必须将 edge 变成一个“有厚度的区域 mask”：

* `M0 = E`（bool）
* `M = dilate(M0, radius=edge_dilate)`
  用 `cv2.dilate` 或 `scipy.ndimage.binary_dilation`

建议默认 `edge_dilate=1`（约 3x3 膨胀一次）。

### 5.3 SDF 构建（标准 mask 的 signed distance）

对二值 mask `M`（True 表示 stroke 区域）构建 signed distance：

* `D_out = distance_transform_edt(~M)`  （在外部点到最近 stroke 的距离）
* `D_in  = distance_transform_edt(M)`   （在内部点到最近背景的距离）
* **Signed distance：**
  [
  \phi = D_{out} - D_{in}
  ]
  性质：
* `phi < 0`：在 stroke 内部
* `phi = 0`：在边界附近
* `phi > 0`：在外部

> 这是真正的 SDF（对 mask 边界而言）。

### 5.4 放大方法 A：Baseline（对 edge/stroke 直接插值）

Baseline 用于对比（必实现，成本低）：

* 输入：`edge_gray` 或 `M.astype(uint8)*255`
* 放大：

  * `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`（bicubic）
  * 或 `INTER_NEAREST`（展示块状锯齿）
* 输出：`baseline_x{scale}.png`

### 5.5 放大方法 B：SDF Upsample（核心）

#### 5.5.1 Upsample SDF

将低分辨率 `phi` 放大到高分辨率：

* `phi_up = cv2.resize(phi, (W*scale, H*scale), interpolation=cv2.INTER_LINEAR)`
* **距离尺度修正**：距离单位随像素缩放

  * `phi_up = phi_up * scale`

#### 5.5.2 从 SDF 渲染高分辨率线稿（抗锯齿）

用一个可控的抗锯齿宽度 `aa_width`（单位：高分辨率像素）把 `phi_up` 映射到 alpha：

定义 smoothstep（必须实现为函数，避免 banding）：

* [
  t = clamp\left(\frac{x-a}{b-a}, 0, 1\right)
  ]
* [
  smoothstep(a,b,x) = t^2(3-2t)
  ]

将 SDF 转 alpha（stroke 内为 1，外为 0，边界平滑）：

* 令 `a = -aa_width`，`b = +aa_width`
* [
  \alpha = 1 - smoothstep(-w, +w, \phi_{up})
  ]
  其中 `w = aa_width`

输出灰度图：

* `out = (alpha * 255).astype(uint8)`

> 这样 SDF upsample 的结果会比直接 bicubic 的 edge 更“几何一致”，线条边界更稳定，锯齿更少。

### 5.6 放大方法 C：PDE（在 Upsampled SDF 上做 PDE 迭代）

这里给出一个足够“PDE 正统”、且实现难度适中的方案：
**先 upsample 得到 `phi_up`，再用 PDE 做两步：**

1. **Reinitialization PDE**：把 `phi_up` 重新拉回“接近距离函数”（满足 (|\nabla \phi| \approx 1)）
2. **Mean Curvature Flow**：对 level set 做曲率平滑，抑制锯齿与小毛刺

#### 5.6.1 Reinitialization PDE（Sussman 风格）

目标 PDE：
[
\phi_t = -\operatorname{sgn}(\phi_0)\left(|\nabla\phi|-1\right)
]

离散化要求（必须）：

* 使用中心差分近似梯度
* 增加 epsilon 防止除零

实现细节：

* `phi0 = phi_up.copy()`
* `sgn = phi0 / sqrt(phi0^2 + eps^2)`（eps 建议 1e-3 或 1e-2）
* 中心差分：

  * `phi_x = (phi[:,2:] - phi[:,:-2]) / 2`（对边界用 pad）
  * `phi_y = (phi[2:,:] - phi[:-2,:]) / 2`
* `grad = sqrt(phi_x^2 + phi_y^2 + eps)`
* 显式迭代：

  * `phi = phi - dt * sgn * (grad - 1)`

参数建议：

* `dt = 0.3`
* `iters_reinit = 20`（demo 足够）

#### 5.6.2 Mean Curvature Flow（对 level set 曲率平滑）

曲率定义：
[
\kappa = \nabla \cdot \left(\frac{\nabla \phi}{|\nabla \phi|}\right)
]

演化 PDE（level set 形式）：
[
\phi_t = \kappa |\nabla \phi|
]

离散化步骤：

1. 计算 `phi_x, phi_y, grad`
2. 单位法向：

   * `nx = phi_x / grad`
   * `ny = phi_y / grad`
3. 散度（中心差分）：

   * `kappa = d(nx)/dx + d(ny)/dy`
4. 更新：

   * `phi = phi + dt * kappa * grad`

参数建议：

* `dt = 0.2`
* `iters_curv = 10`

#### 5.6.3 PDE 输出渲染

对 PDE 后的 `phi` 同样用 5.5.2 的 SDF→alpha 渲染得到 `pde_x{scale}.png`

---

## 6. 质量与参数（必须在前端可调的最小集合）

前端至少提供：

* `scale`：2/4/8 下拉
* `edge_dilate`：0~3 滑条（默认 1）
* `aa_width`：0.5~2.0（默认 1.0）
* （可选）`canny_sigma`：0.2~0.5（默认 0.33）
* methods 勾选（baseline / sdf / pde）

默认值建议：

* scale=4
* canny_sigma=0.33
* edge_dilate=1
* aa_width=1.0
* reinit=20 iters, curvature=10 iters

---

## 7. 前端展示规范（Side-by-side）

### 页面布局（必须）

* 第一行：上传控件 + 参数控件 + “Process” 按钮
* 第二行：图像对比区，至少三列（CSS grid/flex）：

  * Col 1: `Edge (original res)`（或原图+edge）
  * Col 2: `Baseline x{scale}`
  * Col 3: `SDF x{scale}`
  * Col 4: `PDE x{scale}`（如果屏幕窄可以换行，但同方法同大小）

每个结果块：

* 标题（方法名 + scale）
* `<img>` 宽度 100%，`image-rendering` 可选（不建议强制 pixelated，因为我们要展示平滑效果）

### 前端交互（必须）

* 点击 Process 后：

  * disable 按钮
  * 显示 “processing…”（文本即可）
  * 成功返回后更新 `<img src=...>`，带 cache busting（如 `?t=timestamp`）
* 失败时显示错误信息（alert 或页面 div）

---

## 8. 后端实现要点（让 codex 少踩坑）

1. **输入尺寸限制（必须）**
   为避免 scale=8 爆内存：

   * 若输入最大边 > 1024，则按比例缩小到 1024（并在 meta 返回实际处理尺寸）
2. **输出目录**
   `static/out/` 若不存在就创建
3. **文件命名**
   用 `uuid4().hex` 作为 id，输出固定命名，便于前端引用
4. **图像编码**
   统一输出 PNG（灰度 8-bit）
5. **数组边界**
   PDE 中所有差分都需要 pad（reflect/edge），避免尺寸错位
6. **性能**
   PDE 迭代用 numpy 向量化，不要 python 双重 for 循环

---

## 9. requirements.txt（建议）

```
fastapi
uvicorn[standard]
python-multipart
numpy
opencv-python
scipy
```

> 如果实现时想用 `scikit-image`（比如更方便的形态学），可加，但非必须。

---

## 10. README.md（必须包含的运行说明）

* 安装依赖：`pip install -r requirements.txt`
* 运行：`uvicorn app.main:app --reload --port 8000`
* 打开：`http://localhost:8000`
* 说明参数含义与方法差异（Baseline vs SDF vs PDE）

---

## 11. 验收标准（Acceptance Criteria）

1. 打开网页，上传图片，点击 Process，能返回并显示：

   * 原 edge map
   * baseline 放大结果
   * sdf 放大结果
   * pde 放大结果
2. 调整 scale（2/4/8），输出尺寸随之变化，且方法结果不同可见：

   * baseline 相对更糊/锯齿更明显
   * sdf 边界更稳定、抗锯齿更自然
   * pde 在 sdf 基础上减少局部毛刺/锯齿（更平滑）
3. 不崩溃：对 512~1024 边长图片 scale=4 正常输出；scale=8 也能在限制下输出。

---

## 12. 给 codex/gemini-cli 的实现任务清单（可直接粘贴）

1. 创建目录结构与文件（见第 3 节）。
2. 实现 `app/main.py`：

   * FastAPI app
   * `GET /` 返回 index.html
   * mount `/static`
   * `POST /api/process` 接收 multipart，调用 imaging/sdf/pde，保存输出，返回 JSON
3. 实现 `app/imaging.py`：

   * `load_image_from_upload(bytes)->np.ndarray`
   * `resize_if_needed(img,max_side=1024)`
   * `compute_edge_map(gray,sigma)->E_bool, edges_u8`
   * `make_stroke_mask(E_bool, edge_dilate)->M_bool`
4. 实现 `app/sdf.py`：

   * `mask_to_sdf(M_bool)->phi(float32)`
   * `upsample_sdf(phi, scale)->phi_up`
   * `sdf_to_alpha(phi_up, aa_width)->u8`
5. 实现 `app/pde.py`：

   * `reinit_pde(phi_up, dt, iters, eps)->phi`
   * `curvature_flow(phi, dt, iters, eps)->phi`
6. 实现静态前端：

   * `static/index.html`：上传、参数、按钮、结果容器（多列）
   * `static/app.js`：fetch `/api/process`，更新图片
   * `static/style.css`：grid/flex 多列对比
7. README 与 requirements 完整。
