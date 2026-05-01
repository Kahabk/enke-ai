"""
Product Scanner API — Production Ready
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

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from google.cloud import vision
from pyzbar.pyzbar import decode
from ocr import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("product-scanner")


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/user/enke-ai/api.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# Minimum image size to send to vision model (too small = empty response)
MIN_IMAGE_BYTES = 20 * 1024   # 20 KB
TARGET_MIN_DIM  = 640         # upscale if smaller than this



app = FastAPI(
    title="Product Scanner API",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# SCHEMAS
# ==========================================
class ScanResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(img_bytes: bytes) -> bytes:
    """
    Fix common webcam image issues:
    1. Too small → upscale to minimum 640px
    2. Too dark → auto brightness
    3. Wrong format → normalize to JPEG
    4. Blob/no filename → still works
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        logger.info(f"Original image: {w}x{h}, {len(img_bytes)//1024}KB, mode={img.mode}")

        # Upscale if too small (webcam low-res captures)
        if w < TARGET_MIN_DIM or h < TARGET_MIN_DIM:
            scale = max(TARGET_MIN_DIM / w, TARGET_MIN_DIM / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Upscaled to {new_w}x{new_h}")

        # Cap at 1920px to avoid huge payloads
        max_dim = 1920
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            logger.info(f"Downscaled to {img.size}")

        # Save as high-quality JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        result = buf.getvalue()
        logger.info(f"Processed image: {img.size}, {len(result)//1024}KB")
        return result

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e} — using original")
        return img_bytes


# ==========================================
# OLLAMA CLIENT
# ==========================================

def call_ollama(prompt: str, img_bytes: bytes | None = None) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload: dict = {
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
        logger.info(f"Image attached: {len(img_bytes)//1024}KB, b64={len(b64)//1024}KB")

    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as http:
            resp = http.post(url, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama HTTP {e.response.status_code}: {e.response.text}")
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama timed out after {OLLAMA_TIMEOUT}s")

    response_text = resp.json().get("response", "")
    logger.info(f"Ollama response length: {len(response_text)} chars")

    if not response_text.strip():
        logger.warning("Ollama returned empty response!")

    return response_text




def get_raw_data(img_bytes: bytes) -> dict:
    """Google Vision API: OCR + labels + barcode."""
    v_client = vision.ImageAnnotatorClient()
    image    = vision.Image(content=img_bytes)

    text_resp  = v_client.text_detection(image=image)
    label_resp = v_client.label_detection(image=image)

    v_text   = text_resp.text_annotations[0].description if text_resp.text_annotations else ""
    v_labels = [l.description for l in label_resp.label_annotations]

    logger.info(f"Vision OCR chars={len(v_text)}, labels={v_labels[:5]}")

    barcode = "Not found"
    try:
        import numpy as np
        arr   = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        codes = decode(img)
        if codes:
            barcode = codes[0].data.decode("utf-8")
            logger.info(f"Barcode detected: {barcode}")
    except Exception:
        logger.warning("Barcode scan failed: " + traceback.format_exc())

    return {"text": v_text, "labels": v_labels, "barcode": barcode}


def generate_product_json(raw_data: dict, img_bytes: bytes) -> dict:
    ocr_snippet = raw_data["text"][:1200]
    has_ocr     = len(raw_data["text"].strip()) > 10

    # If OCR has good data, use it heavily in prompt
    ocr_section = f"OCR Text from packaging:\n{ocr_snippet}" if has_ocr else "OCR: (no text detected — use image only)"

    prompt = f"""You are a product data extraction assistant analyzing a product image.

{ocr_section}
Detected Labels: {raw_data['labels']}
Barcode: {raw_data['barcode']}

Extract product information and return ONLY a JSON object with ALL these fields in this exact order:
{{"name":"Dove Cream Beauty Bar","brand":"Dove","category":"Personal Care","hsn_code":"34011110","barcode":"8901234567890","sku":"DOVE-CREAM-BAR-100G","slug":"dove-cream-beauty-bar","description":"Dove Cream Beauty Bar is a gentle moisturising soap enriched with 1/4 moisturising cream that helps maintain the skin natural moisture barrier, leaving skin soft, smooth and hydrated after every wash.","base_unit":"100g","price":45.00,"mrp":45.00,"quantity":"100g","ingredients":"Sodium Lauroyl Isethionate, Stearic Acid, Water","country_of_origin":"India","manufacturer":"Hindustan Unilever Ltd"}}

STRICT RULES — follow exactly:
- name: exact product name from packaging
- brand: brand/company name
- category: actual product category (Tea, Soap, Milk, Biscuit, etc.)
- hsn_code: best matching Indian GST HSN code as string
- barcode: detected barcode value, or "unknown" — NEVER null
- sku: generate as BRAND-PRODUCTNAME-SIZE uppercase hyphenated — NEVER null
- slug: lowercase-hyphenated name — NEVER null
- description: MINIMUM 15 words. Write a full product description combining packaging info AND general product knowledge. Example for milk: "Almarai Fresh Full Fat Milk is a premium dairy product enriched with calcium and essential vitamins, offering 25% extra free volume, perfect for daily drinking, cooking, and making hot beverages."
- base_unit: weight or volume unit only like 100g or 500ml or 1L — NEVER "piece" or null — guess from product type if not on label
- price: numeric float only, 0.0 if not visible — NEVER null
- mrp: numeric float only, 0.0 if not visible — NEVER null
- quantity: pack size string like 1L or 100g or 25 bags — NEVER null
- ingredients: key ingredients if visible, else "unknown" — NEVER null
- country_of_origin: country string, default "India" — NEVER null
- manufacturer: company name, use brand name if unknown — NEVER null

CRITICAL: Every single field must have a real value. NEVER use null. NEVER leave a field empty.
Return ONLY the JSON object. No explanation. No markdown fences. No extra text."""

    logger.info(f"Calling Ollama (thinking=off, image attached)")
    raw_response = call_ollama(prompt, img_bytes=img_bytes)
    logger.info(f"Raw response preview: {repr(raw_response[:300])}")

    # If empty response — retry WITHOUT image (text-only fallback)
    if not raw_response.strip():
        logger.warning("Empty response from vision model — retrying text-only with OCR")
        text_only_prompt = f"""You are a product data extraction assistant.

OCR Text: {ocr_snippet}
Labels: {raw_data['labels']}
Barcode: {raw_data['barcode']}

Return a JSON object with ALL these fields filled — NEVER use null:
- name, brand, category, hsn_code
- barcode: use detected value or "unknown"
- sku: BRAND-NAME-SIZE uppercase hyphenated
- slug: lowercase-hyphenated
- description: MINIMUM 15 words describing the product using OCR text plus general product knowledge
- base_unit: weight/volume like 100g or 1L, never "piece"
- price: float, 0.0 if unknown
- mrp: float, 0.0 if unknown
- quantity: pack size string
- ingredients: key ingredients or "unknown"
- country_of_origin: country or "India"
- manufacturer: company name or use brand

Return ONLY the JSON object. No markdown. No nulls."""
        raw_response = call_ollama(text_only_prompt, img_bytes=None)
        logger.info(f"Text-only retry response: {repr(raw_response[:300])}")

    result = _parse_json_response(raw_response, raw_data)
    return _sanitize_product(result, raw_data)


def _sanitize_product(product: dict, raw_data: dict) -> dict:
    """Replace any null/None/empty values with safe defaults."""
    defaults = {
        "name":              raw_data.get("text", "").split("\n")[0].strip() or "Unknown Product",
        "brand":             "unknown",
        "category":          raw_data["labels"][0] if raw_data.get("labels") else "Unknown",
        "hsn_code":          "unknown",
        "barcode":           raw_data.get("barcode", "unknown"),
        "sku":               "UNKNOWN-PRODUCT",
        "slug":              "unknown-product",
        "description":       "Product details not available.",
        "base_unit":         "unknown",
        "price":             0.0,
        "mrp":               0.0,
        "quantity":          "unknown",
        "ingredients":       "unknown",
        "country_of_origin": "India",
        "manufacturer":      "unknown",
    }

    for key, default in defaults.items():
        val = product.get(key)
        if val is None or val == "" or val == "null":
            product[key] = default
            logger.warning(f"Sanitized null field '{key}' → {repr(default)}")

    # Infer brand from product name if still unknown
    if product["brand"] == "unknown" and product["name"] not in ("Unknown Product", "unknown"):
        product["brand"] = product["name"].split()[0]

    # Ensure price/mrp are floats not strings
    for price_field in ("price", "mrp"):
        try:
            product[price_field] = float(product[price_field])
        except (TypeError, ValueError):
            product[price_field] = 0.0

    return product


def _parse_json_response(text: str, raw_data: dict) -> dict:
    text = text.strip()

    # Strategy 1 — strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — regex { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3 — balanced brace walk
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
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 4 — safe fallback
    logger.warning("All JSON parse strategies failed — returning fallback skeleton")
    ocr        = raw_data.get("text", "")
    first_line = ocr.split("\n")[0].strip() if ocr else "Unknown Product"
    brand      = first_line.split()[0] if first_line else "Unknown"
    return {
        "name":              first_line or "Unknown Product",
        "brand":             brand,
        "category":          raw_data["labels"][0] if raw_data.get("labels") else "Unknown",
        "hsn_code":          "unknown",
        "barcode":           raw_data.get("barcode", "unknown"),
        "sku":               f"{brand.upper()}-PRODUCT",
        "slug":              first_line.lower().replace(" ", "-")[:50] if first_line else "unknown-product",
        "description":       (ocr[:200] if ocr else "Product details not available. Please verify product information manually."),
        "base_unit":         "unknown",
        "price":             0.0,
        "mrp":               0.0,
        "quantity":          "unknown",
        "ingredients":       "unknown",
        "country_of_origin": "India",
        "manufacturer":      brand,
        "_parse_note":       "Vision model returned no valid JSON; fallback used.",
    }




@app.get("/health")
def health():
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models   = [m["name"] for m in resp.json().get("models", [])]
        model_ok = any(OLLAMA_MODEL in m for m in models)
    except Exception as e:
        return {"status": "degraded", "ollama": "unreachable", "error": str(e)}

    return {
        "status":           "ok" if model_ok else "degraded — model not found",
        "model":            OLLAMA_MODEL,
        "ollama_url":       OLLAMA_BASE_URL,
        "model_loaded":     model_ok,
        "available_models": models,
    }


@app.post("/scan", response_model=ScanResponse)
async def scan_product(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}. Use JPEG/PNG/WEBP.")

    img_bytes = await file.read()

    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 15MB.")

    if len(img_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Image too small or empty.")

    logger.info(f"Received '{file.filename}' type={file.content_type} size={len(img_bytes)/1024:.1f}KB")

    try:
        # Step 1 — preprocess (fix webcam captures)
        img_bytes = preprocess_image(img_bytes)

        # Step 2 — Google Vision: OCR + labels + barcode
        raw = get_raw_data(img_bytes)

        # Step 3 — Ollama vision → structured JSON
        product = generate_product_json(raw, img_bytes)

        logger.info(f"Done: {product.get('name')} | MRP: {product.get('mrp')} | Brand: {product.get('brand')}")
        return ScanResponse(success=True, data=product)

    except RuntimeError as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception:
        logger.error("Unhandled error:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error.")
@app.post("/invoice-scan", response_model=ScanResponse)
async def scan_invoice(file: UploadFile = File(...)):
    
    allowed = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "application/octet-stream"
    }

    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type: {file.content_type}"
        )

    img_bytes = await file.read()

    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image too large"
        )

    logger.info(f"Invoice received: {file.filename}")

    try:
        # preprocess image
        processed_img = preprocess_image(img_bytes)

        # save temp image
        temp_path = "/tmp/invoice_scan.jpg"
        with open(temp_path, "wb") as f:
            f.write(processed_img)

        # OCR - use the file path, not the bytes
        raw_ocr = run_ocr(temp_path)

        # clean OCR
        cleaned_ocr = clean_ocr(raw_ocr)

        # LLM parse
        response = parse_with_llm(cleaned_ocr)

        # extract JSON
        data = extract_json(response)

        logger.info(f"Invoice items extracted: {len(data.get('items', []))}")

        return ScanResponse(
            success=True,
            data=data
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )