#!/usr/bin/env python3
"""
Ianseo Scorecard Splitter — produce one output folder of club PDFs and clean intermediates.

Behavior when --group-by-club is used:
  - Create temporary single-page PDFs for all inputs
  - Merge pages by club across all inputs into one PDF per club
  - Write all club PDFs into a single output folder
  - Remove all intermediate files and per-input folders

If --group-by-club is not used, script keeps the per-input behavior (creates single-page PDFs
in per-input folders). Use -c/--include-club to prefix filenames with detected club id.

New options:
  -r / --remove-position : redact the eight target position tokens (e.g. 97A/97B/97C/97D/97E/97F/97G/97H) from each page
                           before compression. Uses PyMuPDF to overlay a light-grey rectangle (#E8E8E8).
  -d / --debug : enable detailed debug logging showing which pages were skipped and why
  -j / --jobs : number of parallel processes to use (default: number of CPU cores)

Dependencies:
    sudo apt install ghostscript qpdf
    pip install pikepdf pdfminer.six tqdm pymupdf
"""

import pikepdf
from pdfminer.high_level import extract_text
from tqdm import tqdm
import subprocess
import os
import re
import argparse
from collections import defaultdict
import shutil
import multiprocessing as mp
from functools import partial
# PyMuPDF for robust redaction/overlay
try:
    import fitz  # pymupdf
except Exception:
    fitz = None

# ---------------------------
# Regex helpers
# ---------------------------

# Accept any letter A-H for the position token up to 9999
POSITION_RE = re.compile(r"\b(\d{1,4}[A-H])\b")               # "1A", "97B", "123C", "123E", "1205A"
BARCODE_RE = re.compile(r"(\d{7}[A-Z]-[A-Z]{2}-[A-Za-z]+-\d+)")  # "1234567E-CL-FlBla-1"
CLUB_RE = re.compile(r"\b([0-9A-Z]{7})\b")                # club id: 7 alphanumeric chars (e.g. 1234567)

# ---------------------------
# Extraction functions
# ---------------------------

def extract_info(page_text):
    """
    Extracts (position, barcode) from page_text.
    Returns tuple (position, barcode) or (None, None).
    """
    if not page_text:
        return None, None

    pos_m = POSITION_RE.search(page_text)
    position = pos_m.group(1) if pos_m else None

    bc_m = BARCODE_RE.findall(page_text)
    barcode = bc_m[0] if bc_m else None

    return position, barcode

def extract_club(page_text):
    """
    Extracts a 7-char club id if present in the page text.
    Returns club (string) or None.
    """
    if not page_text:
        return None
    club_m = CLUB_RE.search(page_text)
    return club_m.group(1) if club_m else None

# ---------------------------
# Redaction helper using PyMuPDF
# ---------------------------

def redact_positions_on_pdf(tmp_pdf_path, base_number, fill=(0.91, 0.91, 0.91), pad=0):
    """
    Redact the eight position tokens <base_number>A..H on the single-page PDF at tmp_pdf_path.
    Writes to a temporary file then replaces the original (avoids "save to original must be incremental" error).
    Returns True if any redaction applied, False otherwise.
    """
    if fitz is None:
        print("⚠️ PyMuPDF (pymupdf) not installed — cannot redact positions. Install via: pip install pymupdf")
        return False

    tokens = [f"{base_number}{L}" for L in ("A", "B", "C", "D", "E", "F", "G", "H")]
    edited_any = False

    try:
        doc = fitz.open(tmp_pdf_path)
    except Exception as e:
        print(f"⚠️ Could not open {tmp_pdf_path} with PyMuPDF for redaction: {e}")
        return False

    try:
        for page in doc:
            for token in tokens:
                try:
                    rects = page.search_for(token, quads=False)
                except Exception:
                    rects = []
                if not rects:
                    continue
                for r in rects:
                    rr = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
                    page.add_redact_annot(rr, fill=fill)
                    edited_any = True
            if page.annots():
                try:
                    page.apply_redactions()
                except Exception as e:
                    print(f"⚠️ Failed to apply redactions on a page in {tmp_pdf_path}: {e}")
    finally:
        # always write & close in finally to avoid resource leak
        if edited_any:
            redacted_path = tmp_pdf_path + ".redacted"
            try:
                doc.save(redacted_path, deflate=True)
                doc.close()
                # replace original atomically
                try:
                    os.replace(redacted_path, tmp_pdf_path)
                except Exception as e:
                    print(f"⚠️ Failed to replace original tmp with redacted file: {e}")
                    # keep redacted copy beside original (do not remove)
                    return True
            except Exception as e:
                print(f"⚠️ Failed to save redacted PDF {tmp_pdf_path}: {e}")
                try:
                    doc.close()
                except:
                    pass
                return False
        else:
            try:
                doc.close()
            except:
                pass

    return edited_any

# ---------------------------
# Compression helper (Ghostscript)
# ---------------------------

def compress_pdf_gs(input_path, output_path, color_dpi=300, gray_dpi=300):
    """
    Compress a PDF using Ghostscript with controlled image resolution.
    color_dpi, gray_dpi: ints, DPI for downsampling color and gray images.
    Mono (1-bit) images (barcodes/QR) are kept lossless.
    """
    subprocess.run([
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dDownsampleColorImages=true',
        f'-dColorImageResolution={int(color_dpi)}',
        '-dDownsampleGrayImages=true',
        f'-dGrayImageResolution={int(gray_dpi)}',
        '-dDownsampleMonoImages=false',
        '-dAutoFilterColorImages=true',
        '-dAutoFilterGrayImages=true',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={output_path}',
        input_path
    ], check=True)

# ---------------------------
# Per-input processing (creates single page PDFs)
# ---------------------------

def process_single_input(input_pdf_path, include_club, color_dpi, gray_dpi, output_root, remove_position_flag=False, debug_mode=False):
    """
    Process one input PDF and return:
      - a list of generated single-page file paths
      - a mapping club_id -> list of single-page file paths (club_id may be None)
      - the output directory used for this input
      - debug statistics
    Does NOT perform club grouping; just creates per-page files.
    """
    base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
    if output_root:
        output_dir = os.path.join(output_root, f"{base_name}_pdfs")
    else:
        output_dir = f"{base_name}_pdfs"
    os.makedirs(output_dir, exist_ok=True)

    pdf_in = pikepdf.open(input_pdf_path)
    nb_pages = len(pdf_in.pages)
    print(f"📄 Processing {input_pdf_path} ({nb_pages} pages) …")

    generated_files = []
    club_map = defaultdict(list)  # club_id (str) -> list of file paths
    
    # Debug counters
    debug_stats = {
        'total_pages': nb_pages,
        'skipped_no_position': 0,
        'skipped_no_barcode': 0,
        'skipped_no_both': 0,
        'skipped_extraction_error': 0,
        'skipped_save_error': 0,
        'skipped_compress_error': 0,
        'successful': 0,
        'sample_skipped_pages': []  # Store first 10 skipped pages with details
    }

    # Open debug log file if in debug mode
    debug_log_path = None
    debug_file = None
    if debug_mode:
        debug_log_path = os.path.join(output_dir, f"debug_{base_name}.log")
        debug_file = open(debug_log_path, 'w', encoding='utf-8')
        debug_file.write(f"DEBUG LOG for {input_pdf_path}\n")
        debug_file.write(f"Total pages: {nb_pages}\n")
        debug_file.write("="*80 + "\n\n")

    for i in tqdm(range(nb_pages), desc=f"Processing {base_name}", unit="page"):
        # extract text using pdfminer
        try:
            page_text = extract_text(input_pdf_path, page_numbers=[i])
        except Exception as e:
            if debug_mode and debug_file:
                debug_file.write(f"Page {i+1}: EXTRACTION ERROR - {e}\n")
            debug_stats['skipped_extraction_error'] += 1
            if len(debug_stats['sample_skipped_pages']) < 10:
                debug_stats['sample_skipped_pages'].append({
                    'page': i+1,
                    'reason': f'extraction_error: {e}'
                })
            page_text = ""

        position, barcode = extract_info(page_text)
        detected_club = extract_club(page_text)  # always attempt detection for club grouping

        # Debug logging
        if debug_mode and debug_file:
            debug_file.write(f"Page {i+1}:\n")
            debug_file.write(f"  Position: {position}\n")
            debug_file.write(f"  Barcode: {barcode}\n")
            debug_file.write(f"  Club: {detected_club}\n")
            if not (position and barcode):
                debug_file.write(f"  STATUS: SKIPPED\n")
                # Write first 500 chars of page text for debugging
                debug_file.write(f"  Text sample: {page_text[:500]}\n")
            else:
                debug_file.write(f"  STATUS: OK\n")
            debug_file.write("\n")

        if not (position and barcode):
            # Track specific reasons for skipping
            if not position and not barcode:
                debug_stats['skipped_no_both'] += 1
                reason = 'no_position_and_no_barcode'
            elif not position:
                debug_stats['skipped_no_position'] += 1
                reason = 'no_position'
            else:
                debug_stats['skipped_no_barcode'] += 1
                reason = 'no_barcode'
            
            if len(debug_stats['sample_skipped_pages']) < 10:
                debug_stats['sample_skipped_pages'].append({
                    'page': i+1,
                    'reason': reason,
                    'text_sample': page_text[:200] if page_text else 'NO TEXT'
                })
            continue

        # filename base (without club prefix)
        base_filename = f"{position}_{barcode}.pdf"
        if include_club and detected_club:
            filename = f"{detected_club}_{base_filename}"
        else:
            filename = base_filename

        tmp_path = os.path.join(output_dir, f"tmp_{filename}")
        final_path = os.path.join(output_dir, filename)

        # write single page (uncompressed) using pikepdf
        try:
            pdf_out = pikepdf.Pdf.new()
            pdf_out.pages.append(pdf_in.pages[i])
            pdf_out.save(tmp_path)
        except Exception as e:
            if debug_mode and debug_file:
                debug_file.write(f"Page {i+1}: SAVE ERROR - {e}\n")
            print(f"⚠️ Failed to save temp page {i+1} from {input_pdf_path}: {e}")
            debug_stats['skipped_save_error'] += 1
            continue

        # optionally remove positions BEFORE compression
        if remove_position_flag and position:
            # derive base number from position (e.g. "97A" -> "97")
            m = re.match(r"^(\d{1,4})", position)
            if m:
                base_num = m.group(1)
                redact_positions_on_pdf(tmp_path, base_num, fill=(0.91, 0.91, 0.91), pad=0)
            else:
                # fallback: try to redact the full position token if base num cannot be extracted
                redact_positions_on_pdf(tmp_path, position, fill=(0.91, 0.91, 0.91), pad=0)

        # compress to final
        try:
            compress_pdf_gs(tmp_path, final_path, color_dpi=color_dpi, gray_dpi=gray_dpi)
            os.remove(tmp_path)
        except subprocess.CalledProcessError as e:
            # fallback: keep uncompressed tmp
            if debug_mode and debug_file:
                debug_file.write(f"Page {i+1}: COMPRESS WARNING - {e} (using uncompressed)\n")
            debug_stats['skipped_compress_error'] += 1
            try:
                os.replace(tmp_path, final_path)
            except Exception:
                # if replacement fails, attempt copy
                try:
                    shutil.copy(tmp_path, final_path)
                    os.remove(tmp_path)
                except Exception as e2:
                    print(f"⚠️ Failed to move tmp to final for page {i+1}: {e2}")
                    continue

        generated_files.append(final_path)
        debug_stats['successful'] += 1

        # if group_by_club is desired later, we still collect by detected club (or "unknown")
        key = detected_club if detected_club else "unknown_club"
        club_map[key].append(final_path)

    # Close debug file
    if debug_mode and debug_file:
        debug_file.write("\n" + "="*80 + "\n")
        debug_file.write("SUMMARY:\n")
        debug_file.write(f"Total pages: {debug_stats['total_pages']}\n")
        debug_file.write(f"Successful: {debug_stats['successful']}\n")
        debug_file.write(f"Skipped (no position): {debug_stats['skipped_no_position']}\n")
        debug_file.write(f"Skipped (no barcode): {debug_stats['skipped_no_barcode']}\n")
        debug_file.write(f"Skipped (no both): {debug_stats['skipped_no_both']}\n")
        debug_file.write(f"Skipped (extraction error): {debug_stats['skipped_extraction_error']}\n")
        debug_file.write(f"Skipped (save error): {debug_stats['skipped_save_error']}\n")
        debug_file.write(f"Compress errors (but saved): {debug_stats['skipped_compress_error']}\n")
        debug_file.close()
        if debug_stats['total_pages'] - debug_stats['successful'] > 0:
            print(f"📝 Debug log saved to: {debug_log_path}")

    return generated_files, club_map, output_dir, debug_stats

# ---------------------------
# Club grouping across all inputs (write club PDFs into output directory)
# ---------------------------

def group_clubs_to_folder(all_club_map, clubs_output_dir, color_dpi, gray_dpi):
    """
    all_club_map: mapping club_id -> list of single-page pdf paths (from all inputs)
    clubs_output_dir: directory where club PDFs will be written (will be created)
    This function writes one <club_id>.pdf per club.
    """
    os.makedirs(clubs_output_dir, exist_ok=True)
    club_pdf_paths = []

    for club_id, file_list in all_club_map.items():
        out_pdf = pikepdf.Pdf.new()
        for single_path in file_list:
            try:
                with pikepdf.open(single_path) as spdf:
                    for pg in spdf.pages:
                        out_pdf.pages.append(pg)
            except Exception as e:
                print(f"⚠️ Failed to append {single_path} to club {club_id}: {e}")

        club_pdf_path = os.path.join(clubs_output_dir, f"{club_id}.pdf")
        try:
            out_pdf.save(club_pdf_path)
        except Exception as e:
            print(f"⚠️ Failed to save club PDF for {club_id}: {e}")
            continue

        # compress club pdf (optional but recommended)
        compressed_path = os.path.join(clubs_output_dir, f"{club_id}.compressed.pdf")
        try:
            compress_pdf_gs(club_pdf_path, compressed_path, color_dpi=color_dpi, gray_dpi=gray_dpi)
            os.remove(club_pdf_path)
            os.replace(compressed_path, club_pdf_path)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Ghostscript failed compressing club {club_id}: {e}. Keeping uncompressed club PDF.")

        club_pdf_paths.append(club_pdf_path)

    return club_pdf_paths

# ---------------------------
# Cleanup helper
# ---------------------------

def safe_rmtree(path):
    """
    Remove directory tree if it exists.
    Safety: only removes if path exists and is a directory.
    """
    if path and os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"⚠️ Failed to remove directory {path}: {e}")

# ---------------------------
# Worker function for multiprocessing
# ---------------------------

def process_pdf_worker(pdf_path, include_club, color_dpi, gray_dpi, output_root, remove_position_flag, debug_mode):
    """
    Worker function to process a single PDF file.
    This function is called in parallel by multiple processes.
    """
    try:
        return process_single_input(
            pdf_path,
            include_club=include_club,
            color_dpi=color_dpi,
            gray_dpi=gray_dpi,
            output_root=output_root,
            remove_position_flag=remove_position_flag,
            debug_mode=debug_mode
        )
    except Exception as e:
        print(f"⚠️ Error processing {pdf_path}: {e}")
        return [], {}, None, {
            'total_pages': 0,
            'successful': 0,
            'skipped_no_position': 0,
            'skipped_no_barcode': 0,
            'skipped_no_both': 0,
            'skipped_extraction_error': 0,
            'skipped_save_error': 0,
            'skipped_compress_error': 0,
            'sample_skipped_pages': []
        }

# ---------------------------
# CLI and orchestration
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Split Ianseo multi-page PDFs into single pages and optionally group across all inputs by club (produce one folder of club PDFs).")
    parser.add_argument("pdfs", nargs="+", help="Input PDF files to process.")
    parser.add_argument("-c", "--include-club", action="store_true",
                        help="Prefix filenames with detected club ID (7 chars).")
    parser.add_argument("-g", "--group-by-club", action="store_true",
                        help="Group generated pages by club into per-club PDFs and write them into a single output folder. Removes intermediate files.")
    parser.add_argument("-r", "--remove-position", action="store_true",
                        help="Remove the eight position tokens (e.g. 97A/97B/97C/97D/97E/97F/97G/97H) from each page before saving.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode: create detailed log files showing why pages were skipped.")
    parser.add_argument("-o", "--output-root", default=None,
                        help="Optional root directory to place output folders (defaults to current working directory).")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of parallel jobs (default: number of CPU cores). Use 1 to disable parallelization.")
    parser.add_argument("--color-dpi", type=int, default=300,
                        help="Ghostscript color image DPI for compression (default: 300).")
    parser.add_argument("--gray-dpi", type=int, default=300,
                        help="Ghostscript gray image DPI for compression (default: 300).")

    args = parser.parse_args()

    # Check ghostscript installed
    if not shutil.which('gs'):
        print("⚠️ Warning: Ghostscript 'gs' not found in PATH. Please install ghostscript (sudo apt install ghostscript).")

    # Determine number of parallel jobs
    num_jobs = args.jobs if args.jobs else mp.cpu_count()
    num_jobs = max(1, min(num_jobs, len(args.pdfs)))  # Don't use more processes than PDFs
    
    print(f"🚀 Using {num_jobs} parallel process(es) for {len(args.pdfs)} PDF file(s)")
    print(f"💻 System has {mp.cpu_count()} CPU core(s) available")
    
    # Aggregation containers
    all_generated = []          # list of all generated single-page PDFs
    combined_club_map = defaultdict(list)  # club_id -> list of paths (across all inputs)
    per_input_dirs = []         # track per-input output dirs to clean them later
    all_debug_stats = []        # collect debug stats from all inputs

    # 1) Process each input and create single-page files
    if num_jobs == 1:
        # Sequential processing (no multiprocessing)
        print("📄 Processing files sequentially...")
        for pdf in args.pdfs:
            generated_files, club_map, out_dir, debug_stats = process_pdf_worker(
                pdf,
                include_club=args.include_club,
                color_dpi=args.color_dpi,
                gray_dpi=args.gray_dpi,
                output_root=args.output_root,
                remove_position_flag=args.remove_position,
                debug_mode=args.debug
            )
            
            all_generated.extend(generated_files)
            if out_dir:
                per_input_dirs.append(out_dir)
            all_debug_stats.append((pdf, debug_stats))
            
            # merge club_map into combined_club_map
            for club_id, flist in club_map.items():
                combined_club_map[club_id].extend(flist)
    else:
        # Parallel processing
        print(f"📄 Processing {len(args.pdfs)} files in parallel...")
        
        # Create a partial function with fixed arguments
        worker_func = partial(
            process_pdf_worker,
            include_club=args.include_club,
            color_dpi=args.color_dpi,
            gray_dpi=args.gray_dpi,
            output_root=args.output_root,
            remove_position_flag=args.remove_position,
            debug_mode=args.debug
        )
        
        # Process PDFs in parallel
        with mp.Pool(processes=num_jobs) as pool:
            results = pool.map(worker_func, args.pdfs)
        
        # Aggregate results
        for pdf, (generated_files, club_map, out_dir, debug_stats) in zip(args.pdfs, results):
            all_generated.extend(generated_files)
            if out_dir:
                per_input_dirs.append(out_dir)
            all_debug_stats.append((pdf, debug_stats))
            
            # merge club_map into combined_club_map
            for club_id, flist in club_map.items():
                combined_club_map[club_id].extend(flist)

    if args.group_by_club:
        # Determine clubs output directory (single folder with all club PDFs)
        if args.output_root:
            clubs_output_dir = os.path.join(args.output_root, "club_pdfs")
        else:
            clubs_output_dir = "club_pdfs"
        os.makedirs(clubs_output_dir, exist_ok=True)

        # 2) Create club PDFs across all inputs
        print("\n📦 Grouping pages by club...")
        club_pdf_paths = group_clubs_to_folder(combined_club_map, clubs_output_dir, color_dpi=args.color_dpi, gray_dpi=args.gray_dpi)

        # 3) Cleanup intermediate per-input folders and individual PDFs
        print("🧹 Cleaning up intermediate files...")
        for d in per_input_dirs:
            safe_rmtree(d)

        print(f"\n✅ All club PDFs are in: {os.path.abspath(clubs_output_dir)}")
        print("✅ Intermediate single-page files and per-input folders have been removed.")
        print(f"Total club PDFs created: {len(club_pdf_paths)}")

    else:
        # not grouping: leave individual files in per-input directories
        print("\nNote: --group-by-club not provided. Individual single-page PDFs are kept in per-input folders.")
        print("If you want a single folder with only club PDFs and no intermediates, run with -g/--group-by-club.")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"  - Input files processed: {len(args.pdfs)}")
    print(f"  - Total single-page PDFs produced: {len(all_generated)}")
    if args.group_by_club:
        print(f"  - Clubs grouped across all inputs: {len(combined_club_map)}")
    
    # Print debug summary
    if all_debug_stats:
        print("\nDETAILED STATISTICS PER INPUT:")
        print("-"*80)
        total_pages = 0
        total_successful = 0
        total_skipped = 0
        
        for pdf_name, stats in all_debug_stats:
            total_pages += stats['total_pages']
            total_successful += stats['successful']
            total_skipped += (stats['total_pages'] - stats['successful'])
            
            print(f"\n{pdf_name}:")
            print(f"  Total pages: {stats['total_pages']}")
            print(f"  Successful: {stats['successful']} ({100*stats['successful']/stats['total_pages']:.1f}%)" if stats['total_pages'] > 0 else "  Successful: 0")
            print(f"  Skipped: {stats['total_pages'] - stats['successful']}")
            if stats['skipped_no_position'] > 0:
                print(f"    - No position found: {stats['skipped_no_position']}")
            if stats['skipped_no_barcode'] > 0:
                print(f"    - No barcode found: {stats['skipped_no_barcode']}")
            if stats['skipped_no_both'] > 0:
                print(f"    - No position AND barcode: {stats['skipped_no_both']}")
            if stats['skipped_extraction_error'] > 0:
                print(f"    - Text extraction errors: {stats['skipped_extraction_error']}")
            if stats['skipped_save_error'] > 0:
                print(f"    - Save errors: {stats['skipped_save_error']}")
            
            # Show sample of first skipped pages
            if args.debug and stats['sample_skipped_pages']:
                print(f"\n  First few skipped pages:")
                for skip_info in stats['sample_skipped_pages'][:5]:
                    print(f"    Page {skip_info['page']}: {skip_info['reason']}")
                    if 'text_sample' in skip_info:
                        sample = skip_info['text_sample'].replace('\n', ' ')[:100]
                        print(f"      Text: {sample}...")
        
        print("\n" + "="*80)
        print(f"TOTAL ACROSS ALL FILES:")
        if total_pages > 0:
            print(f"  Pages processed: {total_pages}")
            print(f"  Successful: {total_successful} ({100*total_successful/total_pages:.1f}%)")
            print(f"  Skipped: {total_skipped} ({100*total_skipped/total_pages:.1f}%)")
        print("="*80)
        
        if total_pages > 0 and total_skipped > total_pages * 0.1:  # More than 10% skipped
            print("\n⚠️  WARNING: More than 10% of pages were skipped!")
            if not args.debug:
                print("   Run with --debug flag to generate detailed log files.")
    
    print("\n✅ Processing finished.")

if __name__ == "__main__":
    main()