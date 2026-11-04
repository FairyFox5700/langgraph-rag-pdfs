"""Layout-aware PDF parsing via Azure AI Document Intelligence.

Uses the prebuilt-layout model with markdown output so that:
  - reading order text is preserved,
  - tables are rendered as proper markdown tables (not garbled text), and
  - figures/images are detected with bounding boxes.

Each detected figure is cropped out of the original PDF page (via PyMuPDF)
and sent to a vision-capable Azure OpenAI chat deployment to get a short
text caption, which is appended to the document so images become
searchable/answerable in the RAG pipeline too.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pymupdf as fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import HumanMessage

from . import config

RENDER_DPI = 150
CAPTION_PROMPT = (
    "Describe this figure, chart, diagram, or image extracted from a PDF "
    "document in 2-3 concise sentences. Call out any key data points, axis "
    "labels, trends, or text visible in the image. If it is a photo with no "
    "data content, describe what it depicts."
)


@dataclass
class FigureCaption:
    page_number: int
    caption: str


def _get_client() -> DocumentIntelligenceClient:
    config.require_doc_intel_config()
    return DocumentIntelligenceClient(
        endpoint=config.AZURE_DOC_INTEL_ENDPOINT,
        credential=AzureKeyCredential(config.AZURE_DOC_INTEL_API_KEY),
    )


def _polygon_bbox_inches(polygon: List[float]) -> tuple[float, float, float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _crop_figure_png(pdf_path: Path, page_number: int, polygon: List[float]) -> bytes:
    """Render the given PDF page and crop to the figure's bounding box.

    `page_number` is 1-based (as reported by Document Intelligence).
    `polygon` coordinates are in inches (the unit Document Intelligence
    reports for prebuilt-layout bounding regions).
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        x0, y0, x1, y1 = _polygon_bbox_inches(polygon)
        clip = fitz.IRect(
            int(x0 * RENDER_DPI),
            int(y0 * RENDER_DPI),
            int(x1 * RENDER_DPI),
            int(y1 * RENDER_DPI),
        )
        # Re-render just the clipped region for a crisp crop.
        matrix = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        clip_points = fitz.Rect(x0 * 72, y0 * 72, x1 * 72, y1 * 72)
        cropped = page.get_pixmap(matrix=matrix, clip=clip_points)
        return cropped.tobytes("png")
    finally:
        doc.close()


def _caption_image(chat_model, image_bytes: bytes) -> Optional[str]:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
        response = chat_model.invoke([message])
        return response.content.strip()
    except Exception as exc:  # noqa: BLE001 - captioning is best-effort
        print(f"  Warning: could not caption a figure ({exc}).")
        return None


def analyze_pdf(pdf_path: Path, chat_model=None) -> str:
    """Analyze one PDF and return markdown text (body + figure captions)."""
    client = _get_client()
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=f.read()),
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    result = poller.result()
    markdown = result.content or ""

    figures = getattr(result, "figures", None) or []
    if figures and config.CAPTION_IMAGES and chat_model is not None:
        captions: List[FigureCaption] = []
        for i, figure in enumerate(figures):
            regions = getattr(figure, "bounding_regions", None) or []
            if not regions:
                continue
            region = regions[0]
            try:
                image_bytes = _crop_figure_png(pdf_path, region.page_number, region.polygon)
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: could not crop figure {i} ({exc}).")
                continue
            caption = _caption_image(chat_model, image_bytes)
            if caption:
                captions.append(FigureCaption(page_number=region.page_number, caption=caption))

        if captions:
            markdown += "\n\n## Figure Descriptions\n\n"
            for c in captions:
                markdown += f"- (Page {c.page_number}) {c.caption}\n"

    return markdown
