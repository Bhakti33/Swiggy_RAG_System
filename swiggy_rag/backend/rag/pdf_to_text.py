from pypdf import PdfReader


def pdf_to_text(pdf_path: str) -> str:
    """Extract and clean text from PDF."""
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n".join(pages)

    # Cleaning
    full_text = full_text.replace("\r", "\n")
    full_text = "\n".join(
        [line.strip() for line in full_text.split("\n") if line.strip()]
    )

    return full_text
