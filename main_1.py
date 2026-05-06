"""
Product Scanner API — manully coded so may contain some unused functions and rough edges. Focus on main.py for core logic.
Run: uvicorn main:app --host 0.0.0.0 --port 5000
"""

import os
import cv2
import json
import base64
import logging
import traceback
import re
import io
import time

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from google.cloud import vision
from pyzbar.pyzbar import decode
from ocr import *
# from sarch_imag import *   # not needed – we implement search inline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("product-scanner")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/user/enke-ai/api.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "180"))

MIN_IMAGE_BYTES = 20 * 1024   # 20 KB
TARGET_MIN_DIM  = 640         # upscale if smaller


app = FastAPI(title="Product Scanner API", version="6.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None

def preprocess_image(img_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        logger.info(f"Original image: {w}x{h}, {len(img_bytes)//1024}KB")

        if w < TARGET_MIN_DIM or h < TARGET_MIN_DIM:
            scale = max(TARGET_MIN_DIM / w, TARGET_MIN_DIM / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Upscaled to {new_w}x{new_h}")

        max_dim = 1920
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e} — using original")
        return img_bytes


def get_raw_data(img_bytes: bytes) -> dict:
    v_client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img_bytes)

    text_resp  = v_client.text_detection(image=image)
    label_resp = v_client.label_detection(image=image)

    v_text   = text_resp.text_annotations[0].description if text_resp.text_annotations else ""
    v_labels = [l.description for l in label_resp.label_annotations]

    barcode = "Not found"
    try:
        import numpy as np
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        codes = decode(img)
        if codes:
            barcode = codes[0].data.decode("utf-8")
            logger.info(f"Barcode: {barcode}")
    except Exception:
        logger.warning("Barcode scan failed")

    return {"text": v_text, "labels": v_labels, "barcode": barcode}


def call_ollama(prompt: str, img_bytes: bytes | None = None) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
            "stop": ["Note:", "Explanation:", "---"],
        },
    }
    if img_bytes is not None:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload["images"] = [b64]

    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as http:
            resp = http.post(url, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama timed out after {OLLAMA_TIMEOUT}s")

    return resp.json().get("response", "")


def _parse_json_response(text: str, raw_data: dict) -> dict:
    text = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i+1])
                    except:
                        break
    logger.warning("JSON parse failed – fallback")
    ocr = raw_data.get("text", "")
    first_line = ocr.split("\n")[0].strip() if ocr else "Unknown Product"
    brand = first_line.split()[0] if first_line else "Unknown"
    return {
        "name": first_line or "Unknown Product",
        "brand": brand,
        "category": raw_data["labels"][0] if raw_data.get("labels") else "Unknown",
        "hsn_code": "unknown",
        "barcode": raw_data.get("barcode", "unknown"),
        "sku": f"{brand.upper()}-PRODUCT",
        "slug": first_line.lower().replace(" ", "-")[:50],
        "description": ocr[:200] or "Product details not available.",
        "base_unit": "unknown",
        "price": 0.0,
        "mrp": 0.0,
        "quantity": "unknown",
        "ingredients": "unknown",
        "country_of_origin": "India",
        "manufacturer": brand,
    }

def _sanitize_product(product: dict, raw_data: dict) -> dict:
    defaults = {
        "name": raw_data.get("text", "").split("\n")[0].strip() or "Unknown Product",
        "brand": "unknown",
        "category": raw_data["labels"][0] if raw_data.get("labels") else "Unknown",
        "hsn_code": "unknown",
        "barcode": raw_data.get("barcode", "unknown"),
        "sku": "UNKNOWN-PRODUCT",
        "slug": "unknown-product",
        "description": "Product details not available.",
        "base_unit": "unknown",
        "price": 0.0,
        "mrp": 0.0,
        "quantity": "unknown",
        "ingredients": "unknown",
        "country_of_origin": "India",
        "manufacturer": "unknown",
    }
    for key, default in defaults.items():
        val = product.get(key)
        if val is None or val == "" or val == "null":
            product[key] = default
    if product["brand"] == "unknown" and product["name"] not in ("Unknown Product", "unknown"):
        product["brand"] = product["name"].split()[0]
    for price_field in ("price", "mrp"):
        try:
            product[price_field] = float(product[price_field])
        except:
            product[price_field] = 0.0
    return product

def generate_product_json(raw_data: dict, img_bytes: bytes) -> dict:
    ocr_snippet = raw_data["text"][:1200]
    has_ocr = len(raw_data["text"].strip()) > 10
    ocr_section = f"OCR Text from packaging:\n{ocr_snippet}" if has_ocr else "OCR: (no text detected — use image only)"

    prompt = f"""You are a product data extraction assistant analyzing a product image.

{ocr_section}
Detected Labels: {raw_data['labels']}
Barcode: {raw_data['barcode']}

Extract product information and return ONLY a JSON object with ALL these fields in this exact order:
{{"name":"Dove Cream Beauty Bar","brand":"Dove","category":"Personal Care","hsn_code":"34011110","barcode":"8901234567890","sku":"DOVE-CREAM-BAR-100G","slug":"dove-cream-beauty-bar","description":"Dove Cream Beauty Bar is a gentle moisturising soap...","base_unit":"100g","price":45.00,"mrp":45.00,"quantity":"100g","ingredients":"Sodium Lauroyl Isethionate, Stearic Acid, Water","country_of_origin":"India","manufacturer":"Hindustan Unilever Ltd"}}

STRICT RULES: every field must have a real value. NEVER use null. Return ONLY JSON."""
    raw_response = call_ollama(prompt, img_bytes=img_bytes)
    if not raw_response.strip():
        # fallback text-only
        raw_response = call_ollama(f"Return JSON for product with OCR:\n{ocr_snippet}", img_bytes=None)
    result = _parse_json_response(raw_response, raw_data)
    return _sanitize_product(result, raw_data)

def search_product_image(
    product_name: str,
    max_retries: int = 3
) -> str | None:
    """
    Search DuckDuckGo for a product image.
    Returns IMAGE URL only.
    """

    from ddgs import DDGS

    query = f"{product_name} product"

    for attempt in range(max_retries):

        try:

            with DDGS() as ddgs:

                results = list(
                    ddgs.images(
                        query,
                        max_results=1
                    )
                )

                if not results:

                    logger.warning(
                        f"No images found for '{query}'"
                    )

                    return None

                # GET IMAGE URL
                img_url = results[0].get("image")

                if not img_url:

                    return None

                logger.info(
                    f"Found image URL: {img_url}"
                )

                # RETURN URL ONLY
                return img_url

        except Exception as e:

            err = str(e).lower()

            if (
                "ratelimit" in err or
                "403" in err
            ):

                wait = 2 ** attempt

                logger.info(
                    f"Rate limited, retry in {wait}s"
                )

                time.sleep(wait)

            else:

                logger.error(
                    f"Search error: {e}"
                )

                return None

    return None
@app.get("/scan_by_name", response_model=ScanResponse)
async def scan_product_by_name(product_name: str = Query(..., description="Product name to search")):
    """
    Search for a product image by name, download it, then run the full vision pipeline.
    """
    if not product_name.strip():
        raise HTTPException(status_code=400, detail="product_name is required")
    logger.info(f"Scanning by name: '{product_name}'")

    # 1. Search & download image
    img_bytes = search_product_image(product_name)
    if img_bytes is None:
        raise HTTPException(status_code=404, detail=f"No image found for product '{product_name}'")

    # 2. Process exactly as if uploaded
    try:
        img_bytes = preprocess_image(img_bytes)
        raw = get_raw_data(img_bytes)
        product = generate_product_json(raw, img_bytes)
        logger.info(f"Done: {product.get('name')} | MRP: {product.get('mrp')}")
        return ScanResponse(success=True, data=product)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_ok = any(OLLAMA_MODEL in m for m in models)
    except Exception as e:
        return {"status": "degraded", "ollama": "unreachable", "error": str(e)}
    return {"status": "ok" if model_ok else "degraded", "model": OLLAMA_MODEL}





@app.post("/scan", response_model=ScanResponse)
async def scan_product(
    file: UploadFile = File(...)
):
    allowed = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "application/octet-stream"
    }

    # Validate content type
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type: {file.content_type}"
        )

    # Read uploaded image
    img_bytes = await file.read()

    # Validate size
    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image too large"
        )

    if len(img_bytes) < 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too small"
        )

    logger.info(
        f"Received '{file.filename}' "
        f"type={file.content_type} "
        f"size={len(img_bytes)/1024:.1f}KB"
    )

    try:

        # 1. Preprocess image
        img_bytes = preprocess_image(img_bytes)

        # 2. OCR / Vision
        raw = get_raw_data(img_bytes)

        # 3. Generate structured product JSON
        product = generate_product_json(raw, img_bytes)

        # 4. Get product name
        product_name = product.get("name", "")

        # 5. Search web image URL
        if product_name and product_name != "Unknown Product":

            logger.info(
                f"Searching web image for: {product_name}"
            )

            # IMPORTANT:
            # This function should RETURN IMAGE URL
            # NOT image bytes
            image_url = search_product_image(product_name)

            if image_url:

                logger.info(
                    f"Found image URL: {image_url}"
                )

                # Send URL to Laravel
                product["web_image_url"] = image_url

            else:

                logger.warning(
                    "No web image found"
                )

        else:

            logger.info(
                "No valid product name extracted"
            )

        logger.info(
            f"Done: {product.get('name')} | "
            f"MRP: {product.get('mrp')}"
        )

        return ScanResponse(
            success=True,
            data=product
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception:

        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )
@app.post("/invoice-scan", response_model=ScanResponse)
async def scan_invoice(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    img_bytes = await file.read()
    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    logger.info(f"Invoice received: {file.filename}")
    try:
        processed_img = preprocess_image(img_bytes)
        temp_path = "/tmp/invoice_scan.jpg"
        with open(temp_path, "wb") as f:
            f.write(processed_img)
        raw_ocr = run_ocr(temp_path)
        cleaned_ocr = clean_ocr(raw_ocr)
        response = parse_with_llm(cleaned_ocr)
        data = extract_json(response)
        logger.info(f"Invoice items extracted: {len(data.get('items', []))}")
        return ScanResponse(success=True, data=data)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))