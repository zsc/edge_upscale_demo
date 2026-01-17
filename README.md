# Edge Map Super-Resolution Demo (SDF vs PDE)

A web-based demo comparing two methods for upscaling binary edge maps while maintaining sharpness: **Signed Distance Field (SDF)** and **Shock Filter PDE**.

## Features

*   **Edge Detection**: Uses Canny (auto/manual) with optional Zhang-Suen thinning.
*   **Method 1: SDF Upscaling**: Converts edges to a distance field, upscales the field, and reconstructs edges via thresholding. Guaranteed to be sharp (binary).
*   **Method 2: Shock Filter PDE**: Iteratively evolves a bicubic-interpolated image to sharpen edges using Partial Differential Equations.
*   **Side-by-Side Comparison**: Visually compare Baseline (Bicubic), SDF, and PDE results.
*   **Interactive**: Adjust scale factor, edge width, and detection parameters in the browser.

## Requirements

*   Python 3.10+
*   `pip`

## Installation

```bash
pip install -r backend/requirements.txt
```

## Usage

1.  Start the server:
    ```bash
    uvicorn backend.app:app --reload --port 8000
    ```

2.  Open your browser at:
    `http://127.0.0.1:8000/`

3.  Upload an image (e.g., a line art, icon, or simple photo) and click "Run Upscale".

## Methods Explained

### Baseline (Bicubic)
Standard bicubic interpolation followed by thresholding. Often results in jagged "staircase" artifacts or blurred gray edges if not thresholded hard enough.

### SDF (Signed Distance Field)
Instead of interpolating the pixels directly, we compute the distance from every pixel to the nearest edge. This "distance field" is smooth and continuous. We resize this field and then re-threshold it. This effectively reconstructs the edge at a higher resolution with perfect sharpness.

### PDE (Shock Filter)
Starts with a blurry upscaled image and applies a "Shock Filter". This mathematical process transports information from standard pixels towards the center of edges, creating a discontinuity (shock) at the edge boundary. It effectively "de-blurs" the image numerically.
