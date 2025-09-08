import logging
import time
import base64
from typing import Dict, Generator, Optional, List, Any
import ollama
from ..config import get_settings

logger = logging.getLogger(__name__)


class VisionChatService:
    """Chat service with vision capabilities using LLaVA."""
    
    def __init__(self, text_model: Optional[str] = None, vision_model: str = "llava:7b"):
        settings = get_settings()
        self.text_model = text_model or settings.gen_model
        self.vision_model = vision_model
        self.current_images = []  # Store images for current conversation
        
    def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """Analyze a single image with LLaVA."""
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode()
            
            response = ollama.chat(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64]
                    }
                ]
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def analyze_multiple_images(self, images: List[Dict], prompt: str) -> str:
        """Analyze multiple images from a document."""
        results = []
        
        for idx, img in enumerate(images):
            img_prompt = f"{prompt}\n\nAnalyzing image {idx + 1} from page {img['page']}:"
            # Decode base64 image data
            img_bytes = base64.b64decode(img['data'])
            analysis = self.analyze_image(img_bytes, img_prompt)
            results.append(f"Image {idx + 1} (Page {img['page']}): {analysis}")
        
        return "\n\n".join(results)
    
    def stream_chat_with_context(self, messages: list, pdf_content: Optional[Dict] = None) -> Generator[str, None, Dict]:
        """Stream chat with optional PDF content including images."""
        response_text = ""
        token_count = 0
        start_time = time.time()
        first_token_time = None
        
        try:
            # Check if question is about visual content
            user_message = messages[-1]["content"] if messages else ""
            is_visual_query = any(keyword in user_message.lower() for keyword in [
                "image", "chart", "graph", "diagram", "figure", "picture", "visual",
                "table", "plot", "illustration"
            ])
            
            # If visual query and we have images, use vision model
            if is_visual_query and pdf_content and pdf_content.get("images"):
                # Analyze images first
                image_analysis = self.analyze_multiple_images(
                    pdf_content["images"][:5],  # Limit to first 5 images
                    user_message
                )
                
                # Add image analysis to context
                enhanced_messages = messages[:-1] + [{
                    "role": "system",
                    "content": f"Visual content analysis:\n{image_analysis}"
                }] + [messages[-1]]
                
                model = self.text_model  # Use text model with image analysis context
            else:
                enhanced_messages = messages
                model = self.text_model
            
            # Stream response
            stream = ollama.chat(
                model=model,
                messages=enhanced_messages,
                stream=True
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        response_text += content
                        token_count += len(content.split())
                        yield content
                        
        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            yield f"\nError: {e}"
        
        # Calculate statistics
        end_time = time.time()
        total_time = end_time - start_time
        time_to_first_token = (first_token_time - start_time) if first_token_time else 0
        
        return {
            "response": response_text,
            "token_count": token_count,
            "total_time": total_time,
            "time_to_first_token": time_to_first_token,
            "tokens_per_second": token_count / total_time if total_time > 0 else 0
        }
    
    def describe_document_visuals(self, pdf_content: Dict) -> str:
        """Generate a summary of visual content in a document."""
        summary_parts = []
        
        if pdf_content.get("images"):
            num_images = len(pdf_content["images"])
            summary_parts.append(f"📊 {num_images} image(s) found")
            
            # Analyze first few images for context
            if num_images > 0:
                sample_analysis = self.analyze_image(
                    base64.b64decode(pdf_content["images"][0]["data"]),
                    "Briefly describe what this image shows in one sentence."
                )
                summary_parts.append(f"First image: {sample_analysis}")
        
        if pdf_content.get("tables"):
            num_tables = len(pdf_content["tables"])
            summary_parts.append(f"📋 {num_tables} table(s) found")
        
        return "\n".join(summary_parts) if summary_parts else "No visual content found."