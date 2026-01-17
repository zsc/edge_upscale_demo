写一个python + html demo，实现从一张上传图，计算 edge map，然后用 SDF/ PDE放大（如果有多个方法，在 html 上可以进行 side by side 对比）

---
下面是一份**可以直接交给 gemini-cli / codex 去生成代码**的、可落地的 **Python + HTML demo 规格说明书（spec）**。重点保证：**edge map 超分后仍然锐利**（输出为二值/近二值，避免灰度糊边），并支持 **SDF vs PDE** 在页面上 **side-by-side 对比**。

---

# 1. 项目目标

实现一个 Web Demo：

1. 浏览器上传一张图片（png/jpg）
2. 后端计算 **edge map**（默认 Canny）
3. 对 edge map 做 **放大 / 超分**（至少两种方法）并保证输出仍然“锐利”
4. 前端页面**并排显示**多种方法结果（side-by-side），可调参数并支持下载结果

必须满足：

* **超分后的 edge map 仍然锐利**：

  * 输出应为 **二值(0/255)** 或可选 **轻微抗锯齿但边界非常陡峭**（默认二值）
  * 禁止直接把二值 edge map 用 bilinear/bicubic 放大后当最终结果（那会变灰糊边）
* 至少包含：

  * **SDF 放大法**（Distance Field / Signed Distance Field 思想）
  * **PDE 法**（用 PDE 数值迭代把放大后的灰边“变陡/变二值”，例如 Shock Filter PDE）

---

# 2. 技术选型

## 后端

* Python 3.10+
* FastAPI + Uvicorn（简单易跑）
* OpenCV（`opencv-python` 或 `opencv-contrib-python`）
* NumPy
* Pillow（保存 PNG / 编码 base64）

依赖建议：

* 如果想直接用 thinning：用 `opencv-contrib-python`（带 `cv2.ximgproc.thinning`）
* 如果不想用 contrib：实现一个简化版 Zhang-Suen thinning（可选）

## 前端

* 纯 HTML + CSS + Vanilla JS（不引入框架，方便模型生成）
* `<input type="file">` 上传
* `fetch('/api/process', { method:'POST', body: FormData })`
* 用 CSS Grid 并排展示结果
* 对 edge 图加 `image-rendering: pixelated;` 防止浏览器显示时插值造成“看起来不锐利”

---

# 3. 目录结构（必须产出）

```
edge-sr-demo/
  backend/
    app.py
    edge_pipeline.py
    requirements.txt
  static/
    index.html
    app.js
    style.css
  README.md
```

---

# 4. 运行方式（必须可一键跑起来）

## 安装

```bash
pip install -r backend/requirements.txt
```

## 启动

```bash
uvicorn backend.app:app --reload --port 8000
```

浏览器打开：

* `http://127.0.0.1:8000/`

---

# 5. 前端 UI/交互需求

页面元素（必须有）：

1. 上传图片
2. 参数区（至少）：

   * 放大倍率 `scale`: 2 / 4 / 8（默认 4）
   * Canny 参数：

     * `canny_low`（默认 80）
     * `canny_high`（默认 160）
     * 或 `auto_canny` 开关（默认开；开时忽略 low/high）
   * `edge_width_hr`：输出边线宽度（高分辨率像素单位，默认 1 或 2）
   * `thinning` 开关：是否做细化（默认开，尽量保证线条锐利且细）
3. 结果展示区：至少 3 列并排：

   * Baseline（对照）：`bicubic+threshold`
   * SDF（Distance Field）
   * PDE（Shock Filter）
4. 每列展示：

   * 标题（方法名）
   * 结果图 `<img>`
   * 下载按钮（下载 PNG）
5. 额外展示：

   * 原图
   * 低分辨率 edge map（原尺寸）作为参考

CSS 要求（关键）：

* 对 edge map 的 `<img>` 设置：

  * `image-rendering: pixelated;`
  * 或 `image-rendering: crisp-edges;`（兼容性一般，可加上）
* Grid 布局：三列自适应

---

# 6. 后端 API 设计

## 6.1 `GET /`

* 返回 `static/index.html`

## 6.2 `GET /static/*`

* 静态文件：`app.js`, `style.css`

## 6.3 `POST /api/process`

输入：`multipart/form-data`

字段（必须支持）：

* `file`: 上传图片
* `scale`: int（2/4/8）
* `edge_width_hr`: float（默认 1.5 或 2.0；用于 SDF 重建时的阈值带宽）
* `auto_canny`: bool（默认 true）
* `canny_low`: int（默认 80）
* `canny_high`: int（默认 160）
* `blur_sigma`: float（默认 1.0）
* `thinning`: bool（默认 true）
* `methods`: 逗号分隔字符串（默认 `"baseline,sdf,pde"`）

输出：JSON

```json
{
  "meta": {
    "scale": 4,
    "input_size": [H, W],
    "edge_size": [H, W],
    "output_size": [H*scale, W*scale]
  },
  "images": {
    "original": "data:image/png;base64,...",
    "edge_lr": "data:image/png;base64,...",
    "baseline": "data:image/png;base64,...",
    "sdf": "data:image/png;base64,...",
    "pde": "data:image/png;base64,..."
  }
}
```

注意：

* 输出 edge 相关结果（baseline/sdf/pde）默认应为**二值 PNG**（0/255）
* original 可以原样或缩放后输出 PNG

---

# 7. 核心算法细节（必须照做）

## 7.1 Edge Map 计算（Canny）

输入彩色图 `I`：

1. 转灰度：

* `G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)`

2. 高斯模糊（抑噪）：

* `G_blur = cv2.GaussianBlur(G, ksize=(0,0), sigmaX=blur_sigma)`

3. Canny：

* 若 `auto_canny=True`：

  * `m = median(G_blur)`
  * `low = max(0, (1 - sigma) * m)`，`high = min(255, (1 + sigma) * m)`
  * 推荐 `sigma=0.33`
* 否则使用用户给定的 `canny_low/canny_high`

得到 `E`：uint8，0 或 255。

4. 二值化到 {0,1}：

* `E01 = (E > 0).astype(np.uint8)`

5. （可选但推荐）细化 thinning（保证锐利&细线）

* 若有 `cv2.ximgproc.thinning`：

  * `E_thin = cv2.ximgproc.thinning(E*255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)`
  * `E01 = (E_thin > 0).astype(np.uint8)`
* 否则可选实现 Zhang-Suen thinning（纯 numpy）

输出 `E01`：0/1 edge map（原分辨率）。

---

## 7.2 方法0：Baseline 对照（必须有）

目的：给用户看到“普通放大”为何会糊（哪怕阈值后仍会锯齿/断裂）。

步骤：

1. `U0 = cv2.resize(E01.astype(np.float32), (W*s, H*s), interpolation=cv2.INTER_CUBIC)`
2. 阈值二值化：

   * `E_base = (U0 >= 0.5).astype(np.uint8) * 255`
3. 可选 thinning（默认开）：

   * thinning 后再输出（保证“锐利”但仍作为对照）

> 注意：baseline 仍然可能出现“粗糙/断裂/锯齿”，这是预期对照效果。

---

## 7.3 方法1：SDF / Distance Field 放大（必须实现）

核心思想：
不要直接插值二值边缘，而是：

1. 在低分辨率上计算到“最近边缘像素集合”的距离场 `D`
2. 对距离场插值到高分辨率，并做**尺度修正**
3. 用阈值从距离场重建高分辨率边缘（天然是陡峭边界 → 锐利）

### 7.3.1 距离场定义（针对“线状边缘”用 unsigned distance 即可）

给定 `E01`（edge=1，非edge=0），定义：

* `D_lr(x) = dist(x, {p | E01(p)=1})`

### 7.3.2 用 OpenCV distanceTransform 计算

OpenCV 的 `distanceTransform` 是“到最近 0 像素的距离”。
因此先构造：

* `DT_in = (1 - E01)`，此时：

  * edge 像素为 0
  * 其他为 1

计算：

* `D_lr = cv2.distanceTransform(DT_in, distanceType=cv2.DIST_L2, maskSize=5)`
  输出是 float32，单位是“低分辨率像素”。

### 7.3.3 放大距离场 + 尺度修正（关键）

直接 resize 只是在数值上插值，单位仍是低分辨率像素。
高分辨率像素单位应乘上倍率 `s`：

1. 插值到高分辨率：

* `D_hi_raw = cv2.resize(D_lr, (W*s, H*s), interpolation=cv2.INTER_CUBIC)`

2. 单位修正：

* `D_hi = D_hi_raw * s`

### 7.3.4 从距离场重建“锐利 edge map”

给定输出线宽 `edge_width_hr`（单位：高分辨率像素），定义半径：

* `r = edge_width_hr / 2`

重建二值边缘：

* `E_sdf = (D_hi <= r).astype(np.uint8) * 255`

可选后处理：

* thinning（默认开）：保持线条细且锐
* 或轻微形态学 close（连接断裂）

> 这一步是保证“锐利”的关键：最终输出是阈值后的二值边界，不会产生灰糊边。

---

## 7.4 方法2：PDE（Shock Filter PDE）放大（必须实现）

核心思路：
先得到一个高分辨率灰度边缘概率图 `U0`（例如 bicubic），然后用 PDE 数值迭代把过渡带“变陡”，最后阈值回二值。Shock Filter 的典型效果就是**边界锐化/阶跃化**。

### 7.4.1 初始化

* `U0 = resize(E01 float, bicubic)`，范围 [0,1]
* `U = U0.copy()`

### 7.4.2 Shock Filter PDE（离散形式）

连续形式（直观理解）：

* `u_t = - sign(Δu) * |∇u|`

离散迭代（每步）：

1. 计算梯度（中心差分）：

* `ux = (U[:,2:] - U[:,:-2]) / 2`
* `uy = (U[2:,:] - U[:-2,:]) / 2`
  （边界可用复制 padding 或忽略边缘一圈）

2. 梯度幅值：

* `g = sqrt(ux^2 + uy^2 + eps)`

3. 拉普拉斯（5点模板）：

* `lap = U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - 4*U[i,j]`

4. `sign(lap)` 的平滑近似（避免数值不稳）：

* `sign_lap = lap / sqrt(lap^2 + eps^2)`

5. 更新：

* `U_new = U - dt * sign_lap * g`

6. （推荐）加入保真项，防止漂移：

* `U_new += dt * lambda_fid * (U0 - U)`

  * `lambda_fid` 默认 0.5 ~ 2.0

7. clamp：

* `U = clip(U_new, 0, 1)`

迭代次数：

* `iters` 默认 20（scale=4 时通常够）
* `dt` 默认 0.2
* `eps` 默认 1e-6

### 7.4.3 阈值回二值，保证锐利

* `E_pde = (U >= 0.5).astype(np.uint8) * 255`

可选 thinning（默认开）：

* thinning 保证边缘线细且锐利

> PDE 输出必须二值化，否则会出现灰边。

---

# 8. 参数默认值建议（必须写进代码）

* `scale=4`
* `auto_canny=true`
* `blur_sigma=1.0`
* `edge_width_hr=1.5`（如果 thinning 开启，可以设 2.0 更稳）
* `thinning=true`

PDE 默认：

* `pde_iters=20`
* `pde_dt=0.2`
* `pde_lambda_fid=1.0`
* `pde_eps=1e-6`

---

# 9. 关键“锐利性”验收标准（必须满足）

后端在返回前，必须确保：

1. `baseline/sdf/pde` 三张结果图默认是**二值图**：像素值只包含 `{0, 255}`

   * 可在代码里断言：`np.unique(img).subset({0,255})`
2. SDF 输出不能直接对 edge 图插值作为最终结果，必须“距离场→阈值重建”
3. 前端展示 edge 图时必须加 `image-rendering: pixelated;`，否则视觉上会被浏览器缩放插值“看起来变糊”

---

# 10. 代码模块划分（codex/gemini 生成时的强约束）

## `backend/edge_pipeline.py`

必须提供这些函数（命名可保持一致）：

* `load_image(file_bytes) -> np.ndarray[BGR uint8]`

* `encode_png_base64(img_uint8) -> "data:image/png;base64,..."`

* `compute_edge_map(gray_uint8, auto_canny, canny_low, canny_high, blur_sigma, thinning) -> E01 uint8(0/1)`

* `upscale_baseline(E01, scale, thinning) -> uint8(0/255)`

* `upscale_sdf(E01, scale, edge_width_hr, thinning) -> uint8(0/255)`

* `upscale_pde(E01, scale, iters, dt, lambda_fid, thinning) -> uint8(0/255)`

## `backend/app.py`

* FastAPI app
* mount 静态文件 `/static`
* `GET /` 返回 index.html
* `POST /api/process` 调用 pipeline，返回 JSON（base64 images）

---

# 11. 前端实现要点（避免“看起来不锐利”的坑）

在 `style.css`：

```css
.edge-img {
  image-rendering: pixelated;
  /* 兼容尝试 */
  image-rendering: crisp-edges;
}
```

并且建议：

* edge 图用原始像素尺寸显示或等比缩放（缩放也要 pixelated）
* 每个方法下面加“下载 PNG”按钮：`<a download="sdf.png" href="...">Download</a>`

---

# 12. README 必须包含

* 项目简介（SDF vs PDE）
* 安装与运行命令
* 参数说明
* 方法原理简述（尤其说明为什么 SDF/PDE 可以保持锐利）

---

# 13. 增强

* 增加 `nearest` 放大对照（更像像素风）
* 增加输出：`dist_field_preview`（把距离场可视化）
* 增加一个“edge_width_hr”滑条，实时重建（前端只做显示，重建仍在后端）
* 支持 Sobel/Scharr edge detector 切换
