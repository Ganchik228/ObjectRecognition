from pathlib import Path

from PIL import Image, ImageFilter


def main() -> None:
    image_path = Path(__file__).with_name("Valve_original_(1).PNG")
    image = Image.open(image_path).convert("L")
    edges = image.filter(ImageFilter.FIND_EDGES)
    try:
        edges.show()
    except Exception as exc:
        print(f"Preview skipped: {exc}")


if __name__ == "__main__":
    main()


