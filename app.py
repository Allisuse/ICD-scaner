from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np

app = FastAPI()

class Req(BaseModel):
    image_b64: str
    capacity: int = 1000

@app.post("/analyze")
def analyze(req: Req):

    # decode base64 -> image
    img_data = base64.b64decode(req.image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "bad_image"}

    h, w = img.shape[:2]

    # NOTE: square crop ของคุณมีเส้น 900/100 อยู่ตำแหน่งนี้
    y_hi = int(h * 0.13)   # 900 ml line
    y_lo = int(h * 0.87)   # 100 ml line

    # convert to gray and detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)

    # detect horizontal lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80,
        minLineLength=int(w * 0.35),
        maxLineGap=20
    )

    detected_y = None
    if lines is not None:
        candidates = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(y1 - y2) < 8:
                y = int((y1 + y2) / 2)

                # ต้องอยู่ในช่วง 900-100 เท่านั้น
                if y_hi < y < y_lo:
                    length = abs(x2 - x1)
                    candidates.append((length, y))

        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            detected_y = candidates[0][1]

    if detected_y is None:
        return {"error": "cannot_detect_waterline"}

    # map pixel y -> ml
    # y_hi = 900 ml, y_lo = 100 ml
    ml = 100 + ((y_lo - detected_y) / (y_lo - y_hi)) * 800
    ml = max(0, min(req.capacity, ml))

    pct = int(round((ml / req.capacity) * 100))

    return {
        "fill_ml": int(round(ml)),
        "fill_percent": pct,
        "debug": {
            "y_detect": detected_y,
            "y_900": y_hi,
            "y_100": y_lo
        }
    }
