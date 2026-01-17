import cv2
import numpy as np
import base64
import io
from PIL import Image

def load_image(file_bytes: bytes) -> np.ndarray:
    """Load image from bytes to OpenCV BGR format."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_png_base64(img_uint8: np.ndarray) -> str:
    """Encode OpenCV image to base64 PNG string."""
    # Ensure image is uint8
    if img_uint8.dtype != np.uint8:
        img_uint8 = img_uint8.astype(np.uint8)
    
    # Check if binary (0/1) or (0/255)
    # If 0/1, scale to 0/255 for visibility if needed, but usually we output 0/255
    
    _, buffer = cv2.imencode('.png', img_uint8)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def apply_thinning(img_bin: np.ndarray) -> np.ndarray:
    """Apply Zhang-Suen thinning using opencv-contrib."""
    # Ensure input is 0 or 255 (binary image)
    # cv2.ximgproc.thinning requires 0 for background, 255 for foreground (or non-zero)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(img_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    else:
        # Fallback or simple warning if contrib not installed, though requirements include it.
        # Minimal implementation of thinning is complex, relying on contrib is safer per spec.
        print("Warning: cv2.ximgproc not found. Skipping thinning.")
        return img_bin

def compute_edge_map(
    gray_uint8: np.ndarray,
    auto_canny: bool = True,
    canny_low: int = 80,
    canny_high: int = 160,
    blur_sigma: float = 1.0,
    thinning: bool = True
) -> np.ndarray:
    """
    Compute edge map (0/1).
    Input: Gray image (uint8)
    Output: Binary mask (0/1) of same size.
    """
    # 1. Gaussian Blur
    if blur_sigma > 0:
        # Kernel size 0 means computed from sigma
        g_blur = cv2.GaussianBlur(gray_uint8, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    else:
        g_blur = gray_uint8

    # 2. Canny
    if auto_canny:
        v = np.median(g_blur)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    else:
        lower = canny_low
        upper = canny_high

    edges = cv2.Canny(g_blur, lower, upper)

    # 3. Thinning (Optional but recommended for Canny output consistency)
    if thinning:
        edges = apply_thinning(edges)

    # 4. Binarize to {0, 1}
    # Canny output is 0 or 255.
    e01 = (edges > 0).astype(np.uint8)
    
    return e01

def upscale_baseline(e01: np.ndarray, scale: int, thinning: bool = True) -> np.ndarray:
    """
    Baseline upscale: Bicubic resize -> Threshold.
    Input: e01 (0/1)
    Output: 0/255 uint8
    """
    h, w = e01.shape
    new_h, new_w = h * scale, w * scale
    
    # Resize float
    u0 = cv2.resize(e01.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Threshold at 0.5
    e_base = (u0 >= 0.5).astype(np.uint8) * 255
    
    if thinning:
        e_base = apply_thinning(e_base)
        
    return e_base

def upscale_sdf(e01: np.ndarray, scale: int, edge_width_hr: float, thinning: bool = True) -> np.ndarray:
    """
    SDF Upscale: Distance Transform -> Resize -> Threshold.
    """
    h, w = e01.shape
    new_h, new_w = h * scale, w * scale
    
    # 1. Distance Transform
    # input for distanceTransform: 0 for edge, 1 for background.
    # e01 has 1 for edge.
    dt_in = (1 - e01).astype(np.uint8)
    
    # dist_lr: distance to nearest zero pixel (nearest edge pixel)
    # distanceType=cv2.DIST_L2
    dist_lr = cv2.distanceTransform(dt_in, cv2.DIST_L2, 5)
    
    # 2. Resize Distance Field
    dist_hi_raw = cv2.resize(dist_lr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Scale Correction
    # Since we upscaled spatially by `scale`, the distances in pixel units should also scale.
    dist_hi = dist_hi_raw * scale
    
    # 4. Reconstruct
    # edge_width_hr is the desired width in HR pixels.
    # Points with distance <= radius are part of the edge.
    radius = edge_width_hr / 2.0
    
    e_sdf = (dist_hi <= radius).astype(np.uint8) * 255
    
    if thinning:
        e_sdf = apply_thinning(e_sdf)
        
    return e_sdf

def upscale_pde(
    e01: np.ndarray, 
    scale: int, 
    iters: int = 20, 
    dt: float = 0.2, 
    lambda_fid: float = 1.0, 
    eps: float = 1e-6,
    thinning: bool = True
) -> np.ndarray:
    """
    PDE Upscale: Shock Filter.
    """
    h, w = e01.shape
    new_h, new_w = h * scale, w * scale
    
    # 1. Initialize with Bicubic
    u0 = cv2.resize(e01.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Clip to verify range
    u0 = np.clip(u0, 0, 1)
    
    u = u0.copy()
    
    # 2. Iterations
    for _ in range(iters):
        # Gradients (Central Difference)
        # Pad to handle borders or just slice
        # Using slicing for simplicity:
        # ux[i,j] = (u[i, j+1] - u[i, j-1]) / 2
        # uy[i,j] = (u[i+1, j] - u[i-1, j]) / 2
        
        # We can use cv2.filter2D for gradients or numpy slicing
        # Numpy slicing with padding for boundary conditions
        padded = np.pad(u, 1, mode='edge')
        
        ux = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
        uy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
        
        grad_mag = np.sqrt(ux**2 + uy**2 + eps)
        
        # Laplacian (5-point)
        # lap = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4u[i,j]
        lap = (padded[2:, 1:-1] + padded[:-2, 1:-1] + 
               padded[1:-1, 2:] + padded[1:-1, :-2] - 
               4 * u)
        
        # Sign of Laplacian (Smoothed)
        sign_lap = lap / np.sqrt(lap**2 + eps**2)
        
        # Shock Filter Update
        # u_t = -sign(Lap(u)) * |Grad(u)|
        u_shock = -sign_lap * grad_mag
        
        # Fidelity term: lambda * (u0 - u)
        u_fid = lambda_fid * (u0 - u)
        
        # Total update
        u_new = u + dt * (u_shock + u_fid)
        
        u = np.clip(u_new, 0, 1)
        
    # 3. Binarize
    e_pde = (u >= 0.5).astype(np.uint8) * 255
    
    if thinning:
        e_pde = apply_thinning(e_pde)
        
    return e_pde
