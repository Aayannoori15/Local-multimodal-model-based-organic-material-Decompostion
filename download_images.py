"""Step 1: Download starter images for each decomposition stage from Bing.

Usage:
    python download_images.py            # ~60 images per search query
    python download_images.py --per 40

IMPORTANT: Web images are messy. After downloading, spend 15-20 minutes opening
each dataset/<class>/ folder and deleting wrong images (logos, people, bins with
no waste visible, diagrams, text). Clean data matters more than more data.
"""

import argparse
import hashlib
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image

from config import DATA_DIR

# icrawler's default User-Agent gets an empty result page from Bing, so pose as a browser
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

QUERIES = {
    "1_fresh_waste": [
        "fresh kitchen vegetable peels waste",
        "fresh food scraps compost bin",
        "fresh fruit and vegetable waste pile",
    ],
    "2_early_decomposition": [
        "rotting vegetable scraps mould compost",
        "decomposing food waste early stage",
        "moldy fruit peels compost bin",
    ],
    "3_active_decomposition": [
        "active compost pile partially decomposed",
        "half decomposed compost close up",
        "compost in progress dark with food scraps",
    ],
    "4_mature_compost": [
        "finished mature compost close up",
        "dark crumbly compost humus",
        "ready compost soil in hands",
    ],
}


def clean_folder(folder: Path, min_side: int = 120) -> None:
    """Remove broken, tiny and duplicate images; convert everything to RGB JPG."""
    seen = set()
    kept = 0
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                if min(im.size) < min_side:
                    raise ValueError("too small")
                digest = hashlib.md5(im.resize((32, 32)).tobytes()).hexdigest()
                if digest in seen:
                    raise ValueError("duplicate")
                seen.add(digest)
                out = f.with_suffix(".jpg")
                im.save(out, "JPEG", quality=92)
            if out != f:
                f.unlink()
            kept += 1
        except Exception:
            f.unlink(missing_ok=True)
    print(f"  {folder.name}: {kept} images kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=60, help="images per query")
    args = ap.parse_args()

    for cls, queries in QUERIES.items():
        out = DATA_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(queries):
            print(f"[{cls}] searching: {q}")
            crawler = BingImageCrawler(storage={"root_dir": str(out)}, log_level=40)
            crawler.session.headers.update({"User-Agent": USER_AGENT})
            # file_idx_offset keeps filenames unique across queries
            crawler.crawl(keyword=q, max_num=args.per, file_idx_offset=i * 1000)
        clean_folder(out)

    print("\nDone. Now open each folder in dataset/ and delete wrong images before training.")


if __name__ == "__main__":
    main()
