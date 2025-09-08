#!/usr/bin/env python3
"""Test script for vision capabilities."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processing import VisionDocumentReader
from src.core import VisionChatService


def test_vision_capabilities():
    """Test PDF vision extraction capabilities."""
    print("=" * 50)
    print("Testing Vision Capabilities")
    print("=" * 50)
    
    # Initialize readers
    vision_reader = VisionDocumentReader()
    vision_chat = VisionChatService()
    
    # Check for test PDF
    test_pdf_path = Path("documents") / "test_with_images.pdf"
    
    if not test_pdf_path.exists():
        print("\n⚠️  No test PDF found. Please add a PDF with images to:")
        print(f"   {test_pdf_path}")
        print("\nYou can use any PDF containing:")
        print("  - Charts or graphs")
        print("  - Diagrams or illustrations")
        print("  - Tables")
        return
    
    print(f"\n📄 Testing with: {test_pdf_path}")
    
    # Read PDF with vision
    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    print("\n🔍 Extracting content from PDF...")
    pdf_content = vision_reader.read_pdf_with_vision(pdf_bytes)
    
    # Display extraction results
    print(f"\n✅ Extraction Results:")
    print(f"   - Pages: {pdf_content['metadata']['num_pages']}")
    print(f"   - Images found: {pdf_content['metadata']['num_images']}")
    print(f"   - Tables found: {pdf_content['metadata']['num_tables']}")
    print(f"   - Text length: {len(pdf_content['text'])} characters")
    
    # Test image analysis if images were found
    if pdf_content['images']:
        print("\n🎨 Testing image analysis with LLaVA...")
        first_image = pdf_content['images'][0]
        image_bytes = vision_reader.get_image_for_vision(first_image)
        
        analysis = vision_chat.analyze_image(
            image_bytes,
            "Describe what you see in this image. Is it a chart, diagram, or illustration?"
        )
        
        print(f"\n📊 Image Analysis (Page {first_image['page']}):")
        print(f"   {analysis[:200]}..." if len(analysis) > 200 else f"   {analysis}")
    
    # Test table extraction if tables were found
    if pdf_content['tables']:
        print("\n📋 Tables found:")
        for idx, table in enumerate(pdf_content['tables'][:2]):  # Show first 2 tables
            print(f"\n   Table {idx + 1} (Page {table['page']}):")
            if table['data'] and len(table['data']) > 0:
                # Show headers
                headers = table['data'][0]
                print(f"   Headers: {headers}")
                # Show first data row
                if len(table['data']) > 1:
                    print(f"   First row: {table['data'][1]}")
    
    print("\n✅ Vision capabilities test complete!")
    print("\nThe app can now:")
    print("  • Extract and analyze images from PDFs")
    print("  • Extract tables from PDFs")
    print("  • Answer questions about visual content")
    print("  • Process charts, graphs, and diagrams")


if __name__ == "__main__":
    test_vision_capabilities()