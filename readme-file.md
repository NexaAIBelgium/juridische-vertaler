# Legal Document Translator

A professional legal document translator powered by Google's Gemini 2.0 Flash Lite model. This application provides literal, word-for-word translations while preserving legal formatting and structure.

## Features

- 📄 **Multiple file formats**: PDF (including scanned), TXT
- 🔍 **OCR support**: Automatic text extraction from scanned documents
- 🌍 **14+ languages**: English, Spanish, French, German, Italian, Portuguese, Dutch, Chinese, Japanese, Arabic, Russian, Polish, Korean, Hindi, and custom languages
- ⚖️ **Legal precision**: Literal translations that preserve legal terminology
- 📊 **Smart chunking**: Intelligent document segmentation based on legal structure
- 💾 **Translation caching**: Avoid retranslating identical content
- 🚀 **Streamlit interface**: User-friendly web application

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (for scanned PDFs):
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

3. **Poppler** (for PDF processing):
   - Windows: Download and add to PATH
   - macOS: `brew install poppler`
   - Linux: `sudo apt-get install poppler-utils`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-document-translator.git
cd legal-document-translator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

## Usage

### Streamlit App (Recommended)

```bash
streamlit run legal_translator_app.py
```

Then open your browser to `http://localhost:8501`

### Command Line

```bash
python legal_translator.py
```

## How It Works

1. **Document Analysis**: Detects legal structure (Articles, Sections, Clauses)
2. **Smart Chunking**: Splits documents at natural boundaries
3. **Context Preservation**: Maintains glossaries and defined terms
4. **Literal Translation**: Word-for-word translation preserving legal meaning
5. **Assembly**: Reconstructs the complete translated document

## Configuration

- **API Key**: Set in `.env` file or enter in the app
- **Languages**: Select from 14 pre-configured languages or enter custom
- **Document Type**: Contract, Agreement, Legal Brief, Patent, etc.
- **Advanced Options**: OCR forcing, chunk overlap size

## Example

```python
# Initialize translator
translator = LegalDocumentTranslator(api_key="your-key")

# Translate document
translated = translator.translate_document(
    document_path="contract.pdf",
    source_lang="English",
    target_lang="Spanish",
    document_type="contract"
)
```

## Limitations

- Output limited to 8,192 tokens per chunk
- Rate limits apply (handled automatically)
- Some content may trigger safety filters (original text preserved)

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

- Powered by Google's Gemini 2.0 Flash Lite
- OCR by Tesseract
- PDF processing by PyMuPDF