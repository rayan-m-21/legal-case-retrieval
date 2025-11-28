import re
import pathlib
import os
import shutil

from .config import (
    TRAIN_FILES_DIR, TEST_FILES_DIR,
    CLEAN_TRAIN_FILES_DIR, CLEAN_TEST_FILES_DIR,
)

def clean_legal_text(text):
    """
    Turns the raw cases into clean, embeddable strings
    """

    # <FRAGMENT_SUPPRESSED> segments
    text = re.sub(r"<FRAGMENT_SUPPRESSED>", "", text, flags=re.IGNORECASE)

    # Editor / footer lines (e.g., "Editor: Angela E. McKay/clh" or "Editor: Reginald W. Curtis/blk")
    text = re.sub(r"Editor\s*[:]\s*[^/\n]+[/][^/\n]+", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # End markers (e.g., "[End of document]")
    text = re.sub(r"\[End of document\]", "", text, flags=re.IGNORECASE)

    # Lone page numbers or "Page X" (seen in some files)
    text = re.sub(r"^\[\d+\]$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)

    # strip common court-file headers
    lines = text.splitlines()
    clean_lines = []
    skip = True
    header_patterns = [
        r"Federal Court", r"MLB headnote", r"Indexed As:", r"Counsel:", r"Solicitors of Record:",
        r"Summary:", r"This case is unedited", r"Temp. Cite:", r"Overview", r"Introduction",
        r"A. Introduction", r"OVERVIEW"
    ]
    for line in lines:
        stripped = line.strip()
        # Stop skipping once we hit a line that looks like real content (e.g., [1] or narrative sentence)
        if skip and (stripped.startswith("[") or "." in stripped or ":" in stripped):
            if not any(re.search(pat, stripped, re.IGNORECASE) for pat in header_patterns):
                skip = False
        if skip:
            continue
        clean_lines.append(stripped)

    text = "\n".join(clean_lines)

    # collapsed repeated whitespace
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Normalise quotes, dashes, section symbols
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # En-dash / em-dash → simple hyphen
    text = re.sub(r"[–—]", "-", text)


    # Lines that are *only* a citation like "[2002] F.C.J. No. 980" or "<FRAGMENT_SUPPRESSED> case names"
    text = re.sub(r"^\s*\[[^\]]+\]\s*$", "", text, flags=re.MULTILINE)
    # Remove "supra" or "ibid" if standalone, but keep in sentences
    text = re.sub(r"^\s*(supra|ibid)\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Common endings: "Application dismissed.", "Appeal dismissed.", "JUDGMENT"
    # We trim after the last substantive paragraph
    text = re.sub(r"(Application|Appeal|JUDGMENT|ORDER)\s*(dismissed|allowed)\.?", "", text, flags=re.IGNORECASE)

    # common first lines include _name_ J.
    #   "Tremblay-Lamer, J."
    #   "Rouleau, J."
    #   "Pinard, J."
    # We look for the *first* line that matches the pattern and delete it.
    lines = text.splitlines()
    if lines and re.fullmatch(r".+,\s*J\.", lines[0].strip()):
        # Drop the first line and re-join
        lines = lines[1:]
        text = "\n".join(lines)

    if text.startswith(": ") or text.startswith("[Translation]: "):
        text = re.sub(r"^\[Translation\]: +", "", text).strip()

    text = "\n".join(
        line for line in text.splitlines()
        if "Editor:" not in line and "editor:" not in line.lower()
    )

    # Remove lines that have no alphanumeric characters (i.e., only punctuation, spaces, or nothing)
    text = "\n".join(line for line in text.splitlines() if re.search(r"[a-zA-Z0-9]", line))

    # Convert everything to lowercase
    text = text.lower()

    text = text.strip()

    return text

def clean_dataset(train: bool = True):
    """
    Clean raw legal texts and save to the clean files directory.
    
    Args:
        train: If True, process training files; otherwise process test files.
    """
    raw_dir = TRAIN_FILES_DIR if train else TEST_FILES_DIR
    clean_dir = CLEAN_TRAIN_FILES_DIR if train else CLEAN_TEST_FILES_DIR
    
    # Remove existing clean directory if present
    if clean_dir.is_dir():
        shutil.rmtree(clean_dir)
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    for file in os.listdir(raw_dir):
        path = raw_dir / file
        raw = path.read_text(encoding="utf-8", errors="ignore")
        
        output_path = clean_dir / file
        with open(output_path, 'w') as f:
            print(clean_legal_text(raw), file=f)
    
    print(f"Cleaned {len(os.listdir(clean_dir))} files -> {clean_dir}")


def clean_all_datasets():
    """Clean both train and test datasets."""
    clean_dataset(train=True)
    clean_dataset(train=False)


if __name__ == "__main__":
    clean_all_datasets()