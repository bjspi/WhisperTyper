"""
Utility script to generate macOS .icns and Windows .ico icon files from a single PNG image.

This script is cross-platform (works on Windows, macOS, and Linux) and uses:
- Pillow (PIL) for image resizing
- icnsutil for building the ICNS container
- rich-click for a nicer CLI

Author: Björn
"""

import sys
from pathlib import Path
from typing import List, Optional
import io

from PIL import Image
from icnsutil import IcnsFile
import rich_click as click

def png_to_icns(png_path: str, icns_path: str, keep_iconset: Optional[bool] = False) -> None:
    """
    Convert a square PNG image into a macOS .icns file containing multiple sizes.

    Args:
        png_path (str): Path to the input PNG file. Should be at least 768x768 px.
        icns_path (str): Path to the output ICNS file.
        keep_iconset (bool, optional): If True, also export the resized PNGs
                                       in an .iconset/ folder for inspection.

    Raises:
        FileNotFoundError: If the input PNG does not exist.
        ValueError: If the input image is not square.
    """
    click.echo("--- Generating macOS Icon ---")
    png_file = Path(png_path)
    if not png_file.exists():
        raise FileNotFoundError(f"Input PNG not found: {png_path}")

    img = Image.open(png_file)

    # Check if image is square (macOS icons should be square)
    if img.size[0] != img.size[1]:
        raise ValueError("Input PNG must be square (width == height).")

    # Typical macOS icon sizes (excluding 1024 since source is 768x768)
    sizes: List[int] = [16, 32, 64, 128, 256, 512]

    # All pixel dimensions needed, including standard and retina.
    all_pixel_dims = set()
    for size in sizes:
        all_pixel_dims.add(size)
        all_pixel_dims.add(size * 2)

    # Mapping from pixel size to macOS icon type
    # See: https://en.wikipedia.org/wiki/Apple_Icon_Image_format
    size_to_type = {
        16: "icp4",
        32: "icp5",
        64: "icp6",
        128: "ic07",
        256: "ic08",
        512: "ic09",
        1024: "ic10", # 512x512@2x
    }

    icns = IcnsFile()

    # Optional: iconset folder
    iconset_dir = Path(icns_path).with_suffix(".iconset")
    if keep_iconset:
        iconset_dir.mkdir(exist_ok=True)

    for pixel_dim in sorted(list(all_pixel_dims)):
        # Skip if the required dimension is larger than the source image
        if pixel_dim > img.size[0]:
            continue

        # Skip if there is no corresponding icon type for this dimension
        if pixel_dim not in size_to_type:
            continue

        click.echo(f"⚙️  Generating {pixel_dim}x{pixel_dim}...")
        resized = img.resize((pixel_dim, pixel_dim), Image.LANCZOS)
        with io.BytesIO() as output:
            resized.save(output, format="PNG")
            icns.add_media(key=size_to_type[pixel_dim], data=output.getvalue())

        if keep_iconset:
            # Determine if it's a retina icon for naming in the iconset
            is_retina = any(s * 2 == pixel_dim for s in sizes)
            base_size = pixel_dim // 2 if is_retina else pixel_dim
            suffix = "@2x" if is_retina and base_size in sizes else ""
            icon_name = f"icon_{base_size}x{base_size}{suffix}.png"

            # Avoid overwriting standard icons with retina versions in iconset
            if (iconset_dir / icon_name).exists():
                icon_name = f"icon_{pixel_dim}x{pixel_dim}.png"

            resized.save(iconset_dir / icon_name)


    # Write the ICNS file
    icns.write(icns_path)

    click.secho(f"✅ ICNS created: {icns_path}", fg="green")


def png_to_ico(png_path: str, ico_path: str) -> None:
    """
    Convert a square PNG image into a Windows .ico file.

    Args:
        png_path (str): Path to the input PNG file.
        ico_path (str): Path to the output ICO file.
    """
    click.echo("\n--- Generating Windows Icon ---")
    img = Image.open(png_path)

    if img.size[0] != img.size[1]:
        raise ValueError("Input PNG must be square (width == height).")

    # Standard Windows ICO sizes
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    # Filter sizes that are smaller than or equal to the source image
    valid_sizes = [s for s in ico_sizes if s[0] <= img.size[0]]
    if not valid_sizes:
        raise ValueError(f"Image size ({img.size[0]}x{img.size[0]}) is too small for any ICO sizes.")

    click.echo(f"⚙️  Generating .ico with sizes: {valid_sizes}")
    img.save(ico_path, format='ICO', sizes=valid_sizes)
    click.secho(f"✅ ICO created: {ico_path}", fg="green")


# ------------------ CLI ------------------
@click.command()
@click.argument("input_png", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--keep-iconset",
    is_flag=True,
    default=False,
    help="Also export all resized PNGs into an .iconset/ folder for inspection.",
)
def cli(input_png: str, keep_iconset: bool) -> None:
    """
    Convert a PNG image into macOS .icns and Windows .ico files.

    The output files will be saved in the same directory as the input file.

    INPUT_PNG: Path to the source PNG file (square, e.g. 768x768).
    """
    try:
        # macOS .icns
        output_icns = str(Path(input_png).with_suffix(".icns"))
        png_to_icns(input_png, output_icns, keep_iconset)

        # Windows .ico
        output_ico = str(Path(input_png).with_suffix(".ico"))
        png_to_ico(input_png, output_ico)

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")
        sys.exit(1)

if __name__ == "__main__":
    cli()
