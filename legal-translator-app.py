import streamlit as st
import os
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import google.generativeai as genai
import hashlib
import pickle
from pathlib import Path
from dotenv import load_dotenv
import time
import tempfile
import PyPDF2
import pdf2image
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import base64

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Legal Document Translator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class TranslationChunk:
    """Represents a chunk of text to be translated"""
    id: int
    text: str
    start_pos: int
    end_pos: int
    section_header: Optional[str] = None
    overlap_text: Optional[str] = None

@dataclass
class DocumentMetadata:
    """Stores document-level information for consistent translation"""
    glossary: Dict[str, str]
    defined_terms: Dict[str, str]
    document_type: str
    source_lang: str
    target_lang: str
    jurisdiction: Optional[str] = None
    translation_cache: Dict[str, str] = field(default_factory=dict)

class PDFProcessor:
    """Handles PDF extraction with OCR fallback"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Tuple[str, bool]:
        """
        Extract text from PDF, preserving layout and formatting
        Returns: (text, used_ocr)
        """
        text = ""
        used_ocr = False
        
        try:
            # First try direct text extraction with PyMuPDF
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                # Use get_text with "text" option to preserve layout better
                page_text = page.get_text("text", sort=True)
                
                # Check if page has actual text
                if page_text.strip():
                    text += page_text
                    # Don't add extra newlines - preserve original spacing
                    if page_num < pdf_document.page_count - 1:
                        text += "\n"
                else:
                    # Page seems to be image-based, use OCR
                    used_ocr = True
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # OCR with Tesseract - preserve layout
                    page_text = pytesseract.image_to_string(
                        image, 
                        lang='eng',
                        config='--psm 6'  # Preserve original layout
                    )
                    text += page_text
                    if page_num < pdf_document.page_count - 1:
                        text += "\n"
            
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            # Fallback to PyPDF2
            try:
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for i, page in enumerate(pdf_reader.pages):
                    text += page.extract_text()
                    if i < len(pdf_reader.pages) - 1:
                        text += "\n"
            except:
                # Last resort - full OCR
                used_ocr = True
                text = PDFProcessor.ocr_entire_pdf(pdf_file)
        
        # Don't strip or modify the text - preserve all formatting
        return text, used_ocr
    
    @staticmethod
    def ocr_entire_pdf(pdf_file) -> str:
        """OCR the entire PDF"""
        try:
            pdf_file.seek(0)
            images = pdf2image.convert_from_bytes(pdf_file.read())
            text = ""
            
            progress_bar = st.progress(0)
            for i, image in enumerate(images):
                progress_bar.progress((i + 1) / len(images))
                text += pytesseract.image_to_string(image, lang='eng') + "\n\n"
            
            progress_bar.empty()
            return text
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""

class LegalDocumentTranslator:
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gemini-2.0-flash-lite",
                 cache_dir: str = "./translation_cache"):
        """
        Initialize the translator with Gemini API
        """
        # Configure Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
        # Cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Token limits for Gemini with 70% safety margin
        # 8192 output tokens * 0.7 = 5734 tokens
        # 5734 tokens * 4 chars/token = 22,936 characters
        self.safety_factor = 0.7
        self.max_output_tokens = int(8192 * self.safety_factor)  # 5734 tokens
        self.max_chars_per_chunk = int(self.max_output_tokens * 4)  # ~23,000 chars
        self.absolute_max_chars = 30000  # Hard limit regardless of settings
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation for Gemini)"""
        return len(text) // 4
    
    def detect_document_structure(self, text: str) -> List[Tuple[int, str]]:
        """Detect section headers and structure in legal document"""
        patterns = [
            r'^(?:ARTICLE|Article|SECTION|Section)\s+[IVX\d]+\.?\s*[-–—]?\s*.+$',
            r'^(?:CLAUSE|Clause)\s+\d+\.?\s*[-–—]?\s*.+$',
            r'^\d+\.\s+[A-Z][^.]+$',  # Numbered sections
            r'^[IVX]+\.\s+[A-Z].+$',   # Roman numeral sections
            r'^(?:WHEREAS|NOW THEREFORE|WITNESSETH)',  # Common legal headers
            r'^Schedule\s+\d+|SCHEDULE\s+[A-Z]',  # Schedules
            r'^(?:Exhibit|EXHIBIT)\s+[A-Z\d]',  # Exhibits
        ]
        
        headers = []
        for i, line in enumerate(text.split('\n')):
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                if re.match(pattern, line, re.MULTILINE):
                    pos = text.find(line)
                    if pos != -1:
                        headers.append((pos, line))
                    break
        
        return sorted(headers)
    
    def create_chunks(self, text: str, overlap_size: int = 200, optimal_chunk_size: int = 10000) -> List[TranslationChunk]:
        """Create intelligent chunks based on document structure and size"""
        
        # Calculate optimal number of chunks
        total_length = len(text)
        optimal_chunk_size = 10000  # Target size for better translation quality
        num_chunks = max(1, (total_length + optimal_chunk_size - 1) // optimal_chunk_size)
        target_chunk_size = total_length // num_chunks
        
        print(f"Document length: {total_length} chars")
        print(f"Target chunks: {num_chunks}, ~{target_chunk_size} chars each")
        
        # If document is small, return as single chunk
        if num_chunks == 1:
            return [TranslationChunk(
                id=0,
                text=text,
                start_pos=0,
                end_pos=len(text),
                overlap_text=None,
                section_header="Complete Document"
            )]
        
        # Find all paragraph boundaries (but preserve exact formatting)
        paragraphs = []
        current_pos = 0
        
        # Split by double newlines to find paragraphs, but keep the newlines
        parts = re.split(r'(\n\n+)', text)
        
        for i in range(0, len(parts), 2):
            if i < len(parts):
                para_text = parts[i]
                # Include the separator (newlines) if it exists
                if i + 1 < len(parts):
                    para_text += parts[i + 1]
                
                if para_text:  # Include empty lines too
                    paragraphs.append({
                        'start': current_pos,
                        'end': current_pos + len(para_text),
                        'text': para_text,
                        'length': len(para_text)
                    })
                    current_pos += len(para_text)
        
        # Now distribute paragraphs into chunks
        chunks = []
        chunk_id = 0
        current_chunk_paragraphs = []
        current_chunk_size = 0
        
        for i, para in enumerate(paragraphs):
            # Check if adding this paragraph would make chunk too large
            if current_chunk_size + para['length'] > target_chunk_size * 1.3 and current_chunk_paragraphs:
                # Save current chunk (preserve exact formatting)
                chunk_text = ''.join([p['text'] for p in current_chunk_paragraphs])
                chunk_start = current_chunk_paragraphs[0]['start']
                chunk_end = current_chunk_paragraphs[-1]['end']
                
                overlap_text = None
                if chunks and overlap_size > 0:
                    overlap_text = chunks[-1].text[-overlap_size:]
                
                chunks.append(TranslationChunk(
                    id=chunk_id,
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_end,
                    section_header=f"Part {chunk_id + 1} of {num_chunks}",
                    overlap_text=overlap_text
                ))
                chunk_id += 1
                
                # Start new chunk
                current_chunk_paragraphs = [para]
                current_chunk_size = para['length']
            else:
                # Add to current chunk
                current_chunk_paragraphs.append(para)
                current_chunk_size += para['length'] + 2
        
        # Don't forget the last chunk (preserve exact formatting)
        if current_chunk_paragraphs:
            chunk_text = ''.join([p['text'] for p in current_chunk_paragraphs])
            chunk_start = current_chunk_paragraphs[0]['start']
            chunk_end = current_chunk_paragraphs[-1]['end']
            
            overlap_text = None
            if chunks and overlap_size > 0:
                overlap_text = chunks[-1].text[-overlap_size:]
            
            chunks.append(TranslationChunk(
                id=chunk_id,
                text=chunk_text,
                start_pos=chunk_start,
                end_pos=chunk_end,
                section_header=f"Part {chunk_id + 1} of {num_chunks}",
                overlap_text=overlap_text
            ))
        
        # Rebalance if chunks are very uneven
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        if max_size > min_size * 2:
            print(f"Chunks are unbalanced: {min_size} to {max_size} chars")
            # Could implement rebalancing logic here if needed
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk.text)} chars")
        
        return chunks
    
    def _create_paragraph_chunks(self, text: str, overlap_size: int) -> List[TranslationChunk]:
        """Fallback chunking by paragraphs - improved for legal documents"""
        # Don't split on numbered items that are part of lists
        # Only split on double newlines or major section breaks
        
        chunks = []
        chunk_id = 0
        
        # If the text is small enough, return as single chunk
        if len(text) <= self.max_chars_per_chunk:
            chunks.append(TranslationChunk(
                id=0,
                text=text,
                start_pos=0,
                end_pos=len(text),
                overlap_text=None
            ))
            return chunks
        
        # For larger texts, split more carefully
        # Look for major section breaks (multiple newlines)
        sections = re.split(r'\n{3,}', text)
        
        current_chunk = []
        current_chars = 0
        current_start = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            section_chars = len(section)
            
            # If this single section is too large, we need to split it further
            if section_chars > self.max_chars_per_chunk:
                # Split by paragraphs within the section
                paragraphs = section.split('\n\n')
                for para in paragraphs:
                    para_chars = len(para)
                    if current_chars + para_chars > self.max_chars_per_chunk and current_chunk:
                        # Save current chunk
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(TranslationChunk(
                            id=chunk_id,
                            text=chunk_text,
                            start_pos=current_start,
                            end_pos=current_start + len(chunk_text),
                            overlap_text=chunks[-1].text[-overlap_size:] if chunks else None
                        ))
                        chunk_id += 1
                        current_chunk = [para]
                        current_chars = para_chars
                        current_start += len(chunk_text) + 2
                    else:
                        current_chunk.append(para)
                        current_chars += para_chars + 2
            else:
                # Section fits, add to current chunk or start new one
                if current_chars + section_chars > self.max_chars_per_chunk and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(TranslationChunk(
                        id=chunk_id,
                        text=chunk_text,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text),
                        overlap_text=chunks[-1].text[-overlap_size:] if chunks else None
                    ))
                    chunk_id += 1
                    current_chunk = [section]
                    current_chars = section_chars
                    current_start += len(chunk_text) + 2
                else:
                    if current_chunk:
                        current_chunk.append(section)
                        current_chars += section_chars + 4  # Account for separator
                    else:
                        current_chunk = [section]
                        current_chars = section_chars
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(TranslationChunk(
                id=chunk_id,
                text=chunk_text,
                start_pos=current_start,
                end_pos=len(text),
                overlap_text=chunks[-1].text[-overlap_size:] if chunks else None
            ))
        
        return chunks
    
    def _split_large_section(self, section_text: str, start_pos: int, 
                           header: str, overlap_size: int) -> List[TranslationChunk]:
        """Split a large section into smaller chunks"""
        sub_chunks = self._create_paragraph_chunks(section_text, overlap_size)
        
        for chunk in sub_chunks:
            chunk.start_pos += start_pos
            chunk.end_pos += start_pos
            chunk.section_header = header
        
        return sub_chunks
    
    def extract_legal_terms(self, text: str, source_lang: str, target_lang: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Extract defined terms and create initial glossary"""
        defined_terms = {}
        
        defined_pattern = r'"([^"]+)"\s+(?:means|shall mean|refers to|is defined as)'
        matches = re.finditer(defined_pattern, text, re.IGNORECASE)
        
        for match in matches:
            term = match.group(1)
            def_start = match.end()
            def_end = min(def_start + 200, len(text))
            definition = text[def_start:def_end].split('.')[0].strip()
            defined_terms[term] = definition
        
        # Language-specific glossaries (abbreviated for space)
        glossaries = {
            ("English", "Spanish"): {
                "whereas": "considerando que",
                "hereinafter": "en adelante",
                "pursuant to": "de conformidad con",
                "notwithstanding": "no obstante",
                "force majeure": "fuerza mayor",
            },
            ("English", "French"): {
                "whereas": "attendu que",
                "hereinafter": "ci-après",
                "pursuant to": "conformément à",
                "notwithstanding": "nonobstant",
                "force majeure": "force majeure",
            },
        }
        
        glossary = glossaries.get((source_lang, target_lang), {})
        
        return defined_terms, glossary
    
    def create_translation_prompt(self, chunk: TranslationChunk, 
                                metadata: DocumentMetadata) -> str:
        """Create a detailed prompt for translation"""
        prompt = f"""You are a professional legal translator specializing in {metadata.document_type} documents.
Perform a LITERAL, word-for-word translation from {metadata.source_lang} to {metadata.target_lang}.

CRITICAL TRANSLATION RULES:
1. **LITERAL TRANSLATION**: Translate as literally as possible, word-for-word when feasible
2. **PRESERVE EXACT LAYOUT**: Keep ALL line breaks, spacing, indentation, and formatting EXACTLY as in the original
3. **NO INTERPRETATION**: Do not interpret, paraphrase, or "improve" the text
4. **MAINTAIN FORMATTING**: 
   - If a line has 3 spaces before it, keep 3 spaces
   - If there's a blank line, keep the blank line
   - If text is indented, keep the same indentation
   - Preserve ALL line breaks exactly where they appear
5. **EXACT CORRESPONDENCE**: Each line in the original must have a corresponding line in the translation

CONTEXT:
- Document Type: {metadata.document_type}
- Section: {chunk.section_header or 'General content'}

TEXT TO TRANSLATE (preserve exact formatting):
{chunk.text}

LITERAL TRANSLATION (maintain exact same layout):"""
        
        return prompt
    
    def translate_chunk(self, chunk: TranslationChunk, 
                       metadata: DocumentMetadata) -> str:
        """Translate a single chunk using Gemini API"""
        cache_key = hashlib.md5(
            f"{chunk.text}{metadata.source_lang}{metadata.target_lang}".encode()
        ).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        prompt = self.create_translation_prompt(chunk, metadata)
        
        try:
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
                safety_settings=safety_settings
            )
            
            if response.parts:
                translation = response.text.strip()
            else:
                return f"[TRANSLATION FAILED - Original preserved]\n{chunk.text}"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(translation, f)
            
            return translation
            
        except Exception as e:
            return f"[ERROR: {str(e)}]\n{chunk.text}"

# Streamlit App
def main():
    st.title("⚖️ Legal Document Translator")
    st.markdown("Powered by Gemini 2.0 Flash Lite - Literal translations for legal documents")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        
        # Language selection
        languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", 
                    "Dutch", "Chinese (Simplified)", "Japanese", "Arabic", "Russian", 
                    "Polish", "Korean", "Hindi"]
        
        source_lang = st.selectbox("Source Language", languages, index=0)
        
        # Filter target languages
        target_languages = [lang for lang in languages if lang != source_lang]
        target_lang = st.selectbox("Target Language", target_languages, index=0)
        
        # Document type
        doc_types = ["Contract", "Agreement", "Legal Brief", "Patent", "Terms of Service", "Other"]
        doc_type = st.selectbox("Document Type", doc_types)
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_ocr = st.checkbox("Force OCR (for scanned PDFs)", value=False)
            chunk_overlap = st.slider("Chunk Overlap (characters)", 100, 500, 200)
            
            # Safety margin setting
            safety_margin = st.slider(
                "Output Safety Margin", 
                min_value=50, 
                max_value=90, 
                value=70,
                step=5,
                help="Percentage of output limit to use (70% = conservative, 90% = aggressive)"
            )
            
            # Calculate safe limits based on safety margin
            safe_output_tokens = int(8192 * (safety_margin / 100))
            safe_output_chars = safe_output_tokens * 4
            
            # Determine safe chunk size based on language pair
            if any(lang in [source_lang, target_lang] for lang in ["Chinese (Simplified)", "Japanese", "Korean", "Arabic"]):
                default_chunk = min(10000, safe_output_chars)
                max_chunk = min(15000, safe_output_chars)
                help_text = f"Max {safe_output_chars:,} chars with {safety_margin}% safety margin (Asian/Arabic languages)"
            else:
                default_chunk = min(15000, safe_output_chars)
                max_chunk = min(25000, safe_output_chars)
                help_text = f"Max {safe_output_chars:,} chars with {safety_margin}% safety margin"
            
            optimal_chunk_size = st.slider(
                "Target Chunk Size (characters)", 
                min_value=5000, 
                max_value=max_chunk, 
                value=default_chunk,
                step=1000,
                help=help_text
            )
            
            # Show token estimate with safety margin
            estimated_tokens = optimal_chunk_size // 4
            st.caption(f"≈ {estimated_tokens:,} tokens per chunk")
            st.caption(f"Using {safety_margin}% of {8192:,} token limit = {safe_output_tokens:,} tokens available")
    
    # Main area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['txt', 'pdf', 'docx'],
            help="Upload your legal document (PDF, TXT, or DOCX)"
        )
        
        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                with st.spinner("Extracting text from PDF..."):
                    text, used_ocr = PDFProcessor.extract_text_from_pdf(uploaded_file)
                    if used_ocr:
                        st.info("📸 Used OCR to extract text from images in PDF")
            
            elif uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            
            else:
                st.error("Unsupported file type")
                return
            
            # Display preview
            with st.expander("Preview Original Text"):
                st.text_area("First 1000 characters", text[:1000], height=200)
            
            # Show document statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Document Length", f"{len(text):,} chars")
            with col2:
                st.metric("Estimated Pages", f"~{len(text) // 3000}")
            with col3:
                st.metric("Line Count", f"{text.count(chr(10)):,}")
    
    with col2:
        st.header("🔄 Translation")
        
        if uploaded_file and api_key:
            if st.button("🚀 Start Translation", type="primary"):
                try:
                    # Initialize translator
                    translator = LegalDocumentTranslator(api_key=api_key)
                    
                    # Create metadata
                    defined_terms, glossary = translator.extract_legal_terms(text, source_lang, target_lang)
                    metadata = DocumentMetadata(
                        glossary=glossary,
                        defined_terms=defined_terms,
                        document_type=doc_type.lower(),
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
                    
                    st.info(f"📋 Found {len(defined_terms)} defined terms")
                    
                    # Create chunks
                    chunks = translator.create_chunks(text, chunk_overlap, optimal_chunk_size)
                    st.info(f"✂️ Created {len(chunks)} chunks")
                    
                    # Debug: Show chunk sizes
                    with st.expander("📊 Chunk Information"):
                        total_chars = sum(len(chunk.text) for chunk in chunks)
                        st.write(f"**Total characters in chunks**: {total_chars:,}")
                        st.write(f"**Original text length**: {len(text):,}")
                        
                        if total_chars < len(text) * 0.9:
                            st.error(f"⚠️ WARNING: Only {total_chars/len(text)*100:.1f}% of text is in chunks!")
                        
                        for i, chunk in enumerate(chunks):
                            st.write(f"**Chunk {i+1}**: {len(chunk.text):,} characters")
                            st.write(f"Header: {chunk.section_header}")
                            st.code(f"First 200 chars: {chunk.text[:200]}...")
                            st.code(f"Last 100 chars: ...{chunk.text[-100:]}")
                            st.divider()
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Translate chunks
                    translations = []
                    failed_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        status_text.text(f"Translating chunk {i + 1}/{len(chunks)}...")
                        progress_bar.progress((i + 1) / len(chunks))
                        
                        translation = translator.translate_chunk(chunk, metadata)
                        translations.append(translation)
                        
                        # Check if translation failed
                        if "[TRANSLATION FAILED" in translation or "[ERROR:" in translation:
                            failed_chunks.append(i + 1)
                        
                        # Small delay to avoid rate limits
                        time.sleep(0.5)
                    
                    # Combine translations
                    final_translation = "\n\n".join(translations)
                    
                    status_text.text("✅ Translation complete!")
                    progress_bar.progress(1.0)
                    
                    # Display translation
                    st.subheader("📝 Translated Document")
                    
                    # Show chunk information
                    st.info(f"📊 Document processed in {len(chunks)} chunks")
                    
                    # Display options
                    display_mode = st.radio(
                        "Display mode:",
                        ["Formatted (preserves layout)", "Plain text"],
                        horizontal=True
                    )
                    
                    if display_mode == "Formatted (preserves layout)":
                        # Use code block to preserve spacing and formatting
                        st.code(final_translation, language=None)
                    else:
                        # Regular text area
                        st.text_area("Full Translation", final_translation, height=600)
                    
                    # If there were issues, show them
                    if failed_chunks:
                        st.warning(f"⚠️ Chunks {failed_chunks} could not be translated and were preserved in original form.")
                    
                    # Show individual chunks for debugging
                    with st.expander("🔍 View individual chunks"):
                        for i, (chunk, translation) in enumerate(zip(chunks, translations)):
                            st.write(f"**Chunk {i + 1}** - {chunk.section_header or 'No header'}")
                            st.write("**Original:**")
                            st.text_area(f"Original chunk {i+1}", chunk.text[:500] + "...", height=150, key=f"orig_{i}")
                            st.write("**Translation:**")
                            st.text_area(f"Translated chunk {i+1}", translation[:500] + "...", height=150, key=f"trans_{i}")
                            st.divider()
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Translation",
                        data=final_translation,
                        file_name=f"{uploaded_file.name.split('.')[0]}_translated_{target_lang.lower()}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
        
        elif not uploaded_file:
            st.info("👈 Please upload a document to translate")
        elif not api_key:
            st.warning("👈 Please enter your Gemini API key in the sidebar")

if __name__ == "__main__":
    main()
