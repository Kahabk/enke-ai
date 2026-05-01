import os
import re
import json
import zipfile
import cv2
import numpy as np
import pandas as pd
from xml.sax.saxutils import escape
from google.cloud import vision
from openai import OpenAI

IMAGE_PATH="test1.jpg"
OUTPUT_JSON="output.json"
OUTPUT_EXCEL="output.xlsx"
OPENROUTER_API_KEY="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"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/user/enke-ai/api.json"

client=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

SYSTEM_PROMPT1="""
You are a ledger OCR parser. Extract product rows from OCR invoice text and return structured JSON.

INSTRUCTIONS:
1. Extract ONLY product rows (items with SI number, name, quantity, rate, amount)
2. Ignore: headers, voucher details, totals, footers, party details
3. Merge multiline item names (adjacent lines that continue a product name)
4. Keep original SI numbers
5. Output ONLY valid JSON, no markdown, no explanation

EXPECTED OUTPUT FORMAT:
{
  "items": [
    {
      "SI": "1",
      "Item": "PRODUCT NAME HERE",
      "Qty": "10",
      "Rate": "100.00",
      "Amount": "1000.00"
    }
  ]
}
"""

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy"""
    
    img=cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    denoise=cv2.fastNlMeansDenoising(gray)

    thresh=cv2.threshold(
        denoise,
        0,
        255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )[1]

    resized=cv2.resize(
        thresh,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    kernel=np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]
    ])

    sharpen=cv2.filter2D(
        resized,
        -1,
        kernel
    )

    processed_path="processed.jpg"
    cv2.imwrite(processed_path,sharpen)

    return processed_path

def run_ocr(image_path):
    """Run Google Cloud Vision OCR"""
    
    with open(image_path,"rb") as f:
        img_bytes=f.read()

    v_client=vision.ImageAnnotatorClient()
    image=vision.Image(content=img_bytes)

    text_resp=v_client.document_text_detection(image=image)

    if text_resp.error.message:
        raise Exception(text_resp.error.message)

    text=text_resp.full_text_annotation.text

    return text

def clean_ocr(text):
    """Remove garbage lines and irrelevant text"""
    
    lines=text.split("\n")
    cleaned=[]

    garbage_words=[
        "voucher",
        "party",
        "gross",
        "round",
        "bill amount",
        "net total",
        "sales",
        "return",
        "date",
        "STEP"
    ]

    for line in lines:
        line=line.strip()

        if not line:
            continue

        if any(g in line.lower() for g in garbage_words):
            continue

        cleaned.append(line)

    cleaned_text="\n".join(cleaned)
    cleaned_text=cleaned_text[:6000]  # Increased limit

    return cleaned_text

def parse_with_llm(ocr_text):
    """Parse OCR text using OpenRouter API"""
    
    print("Sending request to OpenRouter...")
    
    completion=client.chat.completions.create(
        model="openai/gpt-4o-mini",  # FIXED: Using reliable model
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT1
            },
            {
                "role":"user",
                "content":f"Parse this OCR invoice text and extract product rows:\n\n{ocr_text}"
            }
        ],
        temperature=0,
        max_tokens=4000
    )

    print("\n=========== API RESPONSE ===========\n")
    print(f"Model: {completion.model}")
    print(f"Stop reason: {completion.choices[0].finish_reason}")
    print(f"Tokens used: {completion.usage.completion_tokens}")
    print("\n====================================\n")

    # Handle response safely
    if not completion.choices or completion.choices[0].message is None:
        raise Exception("OpenRouter returned empty response")

    response = completion.choices[0].message.content

    if response is None:
        raise Exception(
            "OpenRouter returned None content. "
            "The model may not support this request or API key may be invalid."
        )

    return response

def extract_json(response):
    """Extract and validate JSON from response"""
    
    response=response.strip()

    # Remove markdown code blocks if present
    response=response.replace("```json","").replace("```","")

    # Find JSON boundaries
    start=response.find("{")
    end=response.rfind("}")

    if start==-1 or end==-1:
        print("RAW RESPONSE:")
        print(response[:1000])
        raise Exception("No valid JSON found in response")

    json_text=response[start:end+1]

    # Clean up common JSON errors
    json_text=re.sub(r",\s*}", "}", json_text)
    json_text=re.sub(r",\s*]", "]", json_text)

    print("\n=========== EXTRACTED JSON ===========\n")
    print(json_text[:500] + "..." if len(json_text) > 500 else json_text)
    print("\n======================================\n")

    try:
        data=json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"\nJSON Parse Error: {e}")
        print("Full JSON text:")
        print(json_text)
        raise

    return data

def save_json(data):
    """Save data to JSON file"""
    
    with open(OUTPUT_JSON,"w") as f:
        json.dump(data,f,indent=4)
    
    print(f"✓ Saved JSON: {OUTPUT_JSON}")

def excel_column_name(index):
    """Convert a zero-based column index to an Excel column name."""

    name=""
    index+=1

    while index:
        index,remainder=divmod(index-1,26)
        name=chr(65+remainder)+name

    return name

def xlsx_cell(value,row_index,column_index):
    """Create a simple Excel worksheet cell."""

    cell_ref=f"{excel_column_name(column_index)}{row_index}"

    if value is None:
        return f'<c r="{cell_ref}"/>'

    text=str(value)

    if re.fullmatch(r"-?(?:\d+\.?\d*|\.\d+)",text):
        number=text

        if number.startswith("."):
            number="0"+number
        elif number.startswith("-."):
            number="-0"+number[1:]

        return f'<c r="{cell_ref}"><v>{number}</v></c>'

    safe_text=escape(text)
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{safe_text}</t></is></c>'

def write_xlsx(df,path):
    """Write a basic .xlsx workbook without external Excel dependencies."""

    rows=[list(df.columns)]+df.values.tolist()
    sheet_rows=[]

    for row_index,row in enumerate(rows,start=1):
        cells=[
            xlsx_cell(value,row_index,column_index)
            for column_index,value in enumerate(row)
        ]
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    last_cell=f"{excel_column_name(len(df.columns)-1)}{len(rows)}"
    worksheet_xml=f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{last_cell}"/>
  <sheetData>
    {"".join(sheet_rows)}
  </sheetData>
</worksheet>'''

    workbook_xml='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="OCR Output" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''

    workbook_rels='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    root_rels='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    content_types='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

    styles_xml='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>'''

    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml",content_types)
        workbook.writestr("_rels/.rels",root_rels)
        workbook.writestr("xl/workbook.xml",workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels",workbook_rels)
        workbook.writestr("xl/worksheets/sheet1.xml",worksheet_xml)
        workbook.writestr("xl/styles.xml",styles_xml)

def save_excel(data):
    """Save data to Excel file"""
    
    items=data.get("items",[])

    if not items:
        print("WARNING: No items to save to Excel")
        return

    df=pd.DataFrame(items)

    print("\n=========== DATA PREVIEW ===========\n")
    print(df)
    print(f"\nTotal items extracted: {len(items)}")
    print("\n====================================\n")

    write_xlsx(df,OUTPUT_EXCEL)
    print(f"✓ Saved Excel: {OUTPUT_EXCEL}")

def main():
    """Main execution flow"""
    
    try:
        print("="*50)
        print("OCR INVOICE PARSER - FIXED VERSION")
        print("="*50)
        
        print("\n[1/7] PREPROCESSING IMAGE")
        processed_path=preprocess_image(IMAGE_PATH)
        print("✓ Image preprocessed")

        print("\n[2/7] RUNNING OCR")
        raw_ocr=run_ocr(processed_path)
        print(f"✓ OCR completed ({len(raw_ocr)} chars)")

        print("\n[3/7] CLEANING OCR TEXT")
        cleaned_ocr=clean_ocr(raw_ocr)
        print(f"✓ Text cleaned ({len(cleaned_ocr)} chars)")

        print("\n[4/7] PARSING WITH LLM")
        response=parse_with_llm(cleaned_ocr)

        print("\n[5/7] EXTRACTING JSON")
        data=extract_json(response)
        print(f"✓ Extracted {len(data.get('items',[]))} items")

        print("\n[6/7] SAVING JSON")
        save_json(data)

        print("\n[7/7] SAVING Excel")
        save_excel(data)

        print("\n" + "="*50)
        print("✓ DONE - All files saved successfully!")
        print("="*50)

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check IMAGE_PATH and file exists")
        print("2. Verify Google Cloud credentials at api.json")
        print("3. Verify OpenRouter API key is valid")
        print("4. Check if account has sufficient credits")
        raise

if __name__=="__main__":
    main()
