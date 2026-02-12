"""
OCR Processor Module
Handles OCR extraction, watermark removal, and text preprocessing
"""

import os
import re
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes, convert_from_path
    import PyPDF2
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install pytesseract pdf2image PyPDF2 Pillow")


class OCRProcessor:
    """
    OCR Processor for extracting text from PDF and image files
    """
    
    def __init__(self, engine: str = "tesseract", config: Optional[Dict[str, Any]] = None):
        """
        Initialize OCR Processor
        
        Args:
            engine: OCR engine to use ('tesseract' or 'nanonets')
            config: Additional configuration parameters
        """
        self.engine = engine.lower()
        self.config = config or {}
        
        # Set Tesseract path for Windows
        tesseract_path = os.getenv('TESSERACT_PATH')
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Watermark patterns to remove
        self.watermark_patterns = [
            r'(?i)copyright\s*©?\s*\d{4}',
            r'(?i)confidential',
            r'(?i)draft',
            r'(?i)watermark',
            r'(?i)private\s+and\s+confidential',
            r'(?i)internal\s+use\s+only',
            r'(?i)do\s+not\s+copy',
        ]
    
    def extract_text(self, file, file_type: Optional[str] = None) -> str:
        """
        Extract text from uploaded file
        
        Args:
            file: Uploaded file object (Streamlit UploadedFile or file path)
            file_type: Type of file ('pdf', 'image', or None for auto-detect)
        
        Returns:
            Extracted text string
        """
        if isinstance(file, str) or isinstance(file, Path):
            # File path provided
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type from extension
            if file_type is None:
                ext = file_path.suffix.lower()
                if ext == '.pdf':
                    file_type = 'pdf'
                elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    file_type = 'image'
            
            if file_type == 'pdf':
                return self._extract_from_pdf_path(file_path)
            else:
                return self._extract_from_image_path(file_path)
        else:
            # Streamlit UploadedFile object
            if file_type is None:
                file_type = 'pdf' if file.type == 'application/pdf' else 'image'
            
            if file_type == 'pdf':
                return self._extract_from_pdf_bytes(file.read())
            else:
                return self._extract_from_image_bytes(file.read())
    
    def _extract_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        text = ""
        
        try:
            # First, try to extract text directly from PDF
            pdf_file = PyPDF2.PdfReader(tempfile.NamedTemporaryFile(delete=False))
            
            # Alternative: Use BytesIO
            import io
            pdf_file = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            
            # Extract text from all pages
            for page_num in range(len(pdf_file.pages)):
                page = pdf_file.pages[page_num]
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
            
            # If no text extracted, use OCR on images
            if len(text.strip()) < 100:
                text = self._ocr_from_pdf_bytes(pdf_bytes)
        
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            # Fallback to OCR
            text = self._ocr_from_pdf_bytes(pdf_bytes)
        
        return text
    
    def _extract_from_pdf_path(self, pdf_path: Path) -> str:
        """Extract text from PDF file path"""
        with open(pdf_path, 'rb') as f:
            return self._extract_from_pdf_bytes(f.read())
    
    def _ocr_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Perform OCR on PDF by converting to images"""
        text = ""
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            
            # Perform OCR on each page
            for i, image in enumerate(images):
                page_text = self._perform_ocr(image)
                text += f"\n[Page {i+1}]\n{page_text}\n"
        
        except Exception as e:
            print(f"Error performing OCR on PDF: {e}")
        
        return text
    
    def _extract_from_image_bytes(self, image_bytes: bytes) -> str:
        """Extract text from image bytes"""
        try:
            import io
            image = Image.open(io.BytesIO(image_bytes))
            return self._perform_ocr(image)
        except Exception as e:
            print(f"Error extracting from image: {e}")
            return ""
    
    def _extract_from_image_path(self, image_path: Path) -> str:
        """Extract text from image file path"""
        try:
            image = Image.open(image_path)
            return self._perform_ocr(image)
        except Exception as e:
            print(f"Error extracting from image: {e}")
            return ""
    
    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on an image using selected engine"""
        if self.engine == "tesseract":
            return self._tesseract_ocr(image)
        elif self.engine == "nanonets":
            return self._nanonets_ocr(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
    
    def _tesseract_ocr(self, image: Image.Image) -> str:
        """Perform OCR using Tesseract"""
        try:
            # Tesseract configuration
            custom_config = self.config.get('tesseract_config', '--psm 6')
            
            # Perform OCR
            text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
            return text
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed: {e}")
    
    def _nanonets_ocr(self, image: Image.Image) -> str:
        """Perform OCR using Nanonets API (placeholder)"""
        # TODO: Implement Nanonets API integration
        raise NotImplementedError("Nanonets OCR not implemented yet. Use 'tesseract' instead.")
    
    def remove_watermarks(self, text: str) -> str:
        """
        Remove watermarks and unwanted patterns from text
        
        Args:
            text: Input text with potential watermarks
        
        Returns:
            Cleaned text
        """
        cleaned_text = text
        
        # Remove watermark patterns
        for pattern in self.watermark_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text
        
        Args:
            text: Raw extracted text
        
        Returns:
            Preprocessed text
        """
        # Remove watermarks
        text = self.remove_watermarks(text)
        
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove page numbers
        text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
        
        # Remove headers/footers (common patterns)
        text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
        
        # Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors"""
        # Common OCR character replacements
        replacements = {
            'l': 'I',  # lowercase L to uppercase I in certain contexts
            '0': 'O',  # zero to O in certain contexts
            '§': 'S',
            '©': '',
            '®': '',
            '™': '',
        }
        
        # Apply context-aware fixes
        # (This is simplified; you can add more sophisticated logic)
        
        return text
    
    def extract_metadata(self, file) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            file: Uploaded file object
        
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'filename': getattr(file, 'name', 'unknown'),
            'size': getattr(file, 'size', 0),
            'type': getattr(file, 'type', 'unknown'),
        }
        
        return metadata


# Standalone functions for easy import
def extract_text_from_file(file, engine: str = "tesseract") -> str:
    """
    Convenience function to extract text from file
    
    Args:
        file: File object or path
        engine: OCR engine to use
    
    Returns:
        Extracted text
    """
    processor = OCRProcessor(engine=engine)
    text = processor.extract_text(file)
    return processor.preprocess_text(text)
