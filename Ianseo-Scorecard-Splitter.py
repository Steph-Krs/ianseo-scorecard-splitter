import pikepdf
from pdfminer.high_level import extract_text
from tqdm import tqdm
import subprocess
import os
import re
import zipfile
import sys

def extract_info(page_text):
    """
    Extracts the position and barcode from the page text.
    Returns the filename to use for the page, or None if not found.
    """
    position_match = re.search(r"\\b(\\d{1,3}A)\\b", page_text)
    position = position_match.group(1) if position_match else None

    barcode_matches = re.findall(r"(\\d{7}[A-Z]-[A-Z]{2}-[A-Za-z]+-\\d)", page_text)
    barcode = barcode_matches[0] if barcode_matches else None

    if position and barcode:
        return f"{position}_{barcode}.pdf"
    return None

def compress_pdf(input_path, output_path):
    """
    Compresses a PDF file using Ghostscript while preserving image quality (~300dpi).
    """
    subprocess.run([
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dDownsampleColorImages=true',
        '-dColorImageResolution=300',
        '-dDownsampleGrayImages=true',
        '-dGrayImageResolution=300',
        '-dDownsampleMonoImages=false',
        '-dAutoFilterColorImages=true',
        '-dAutoFilterGrayImages=true',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={output_path}',
        input_path
    ], check=True)

def process_pdf(input_pdf_path, zip_output_path):
    """
    Main function: processes a multi-page PDF, splits into single-page PDFs with unique names,
    compresses each page, and archives them into a single ZIP file.
    """
    pdf_in = pikepdf.open(input_pdf_path)
    nb_pages = len(pdf_in.pages)
    output_dir = f"{os.path.splitext(os.path.basename(input_pdf_path))[0]}_pdfs"
    os.makedirs(output_dir, exist_ok=True)

    written_files = []

    print(f"📄 Processing {input_pdf_path} ({nb_pages} pages) …")


    for i in tqdm(range(nb_pages), desc="Progress", unit="page"):
        text = extract_text(input_pdf_path, page_numbers=[i])
        filename = extract_info(text)
        if filename:
            tmp_path = os.path.join(output_dir, f"tmp_{filename}")
            final_path = os.path.join(output_dir, filename)

            pdf_out = pikepdf.Pdf.new()
            pdf_out.pages.append(pdf_in.pages[i])
            pdf_out.save(tmp_path)

            # Compress with Ghostscript and replace the temporary file
            compress_pdf(tmp_path, final_path)
            os.remove(tmp_path)

            written_files.append(final_path)

    if not written_files:
        print(f"⚠️ No valid pages were extracted from {input_pdf_path}")
        return

    with zipfile.ZipFile(zip_output_path, "w") as zipf:
        for file_path in written_files:
            arcname = os.path.basename(file_path)
            zipf.write(file_path, arcname=arcname)

    print(f"✅ Archive created: {zip_output_path}")
    print(f"🗑️ You can delete the temporary folder {output_dir} after verification.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Ianseo-Scorecard-Splitter.py file1.pdf file2.pdf …")
        sys.exit(1)

    for pdf in sys.argv[1:]:
        zip_name = f"{os.path.splitext(os.path.basename(pdf))[0]}.zip"
        process_pdf(pdf, zip_name)