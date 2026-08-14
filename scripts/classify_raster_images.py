import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


EXPECTED_IMAGE_COUNT = 753

MODEL_NAME = "openai/clip-vit-base-patch32"

ORIGINAL_REQUIRED_FIELDS = {
    "image_id",
    "digest",
    "pixel_width",
    "pixel_height",
    "file_path",
    "file_format",
    "occurrence_count",
    "first_pdf_page",
    "classification_status",
}

CLASSIFICATION_FIELDS = [
    "predicted_type",
    "classification_confidence",
    "classification_margin",
    "classification_method",
    "review_status",
    "final_type",
    "content_relevance",
]

CLASS_PROMPTS = {
    "microscopy": [
        "a scientific microscopy image",
        "a microscopic laboratory specimen",
        "cells or microorganisms viewed under a microscope",
        "a medical microscopy photograph",
    ],
    "clinical_or_laboratory": [
        "a clinical laboratory photograph",
        "medical laboratory equipment or material",
        "a diagnostic specimen or laboratory procedure",
        "a scientific clinical photograph",
    ],
    "diagram_or_chart": [
        "a scientific diagram or chart",
        "a flowchart or technical illustration",
        "a graph with plotted scientific data",
        "an explanatory medical diagram",
    ],
    "table_or_form": [
        "a scanned table with rows and columns",
        "a document form",
        "a table containing text and numbers",
        "a structured checklist or laboratory form",
    ],
    "document_element": [
        "a document icon or page element",
        "a small symbol from a printed document",
        "a document heading ornament or marker",
        "a functional graphical element in a manual",
    ],
    "decorative_or_logo": [
        "a logo or decorative graphic",
        "an organization emblem",
        "a decorative page illustration",
        "a branding element",
    ],
    "fragment_or_noise": [
        "a tiny meaningless image fragment",
        "a blank or nearly blank image",
        "a scanning artifact or visual noise",
        "an incomplete cropped graphical fragment",
    ],
}


def configure_csv_field_limit() -> None:
    maximum_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maximum_limit)
            return
        except OverflowError:
            maximum_limit //= 10


def read_images_csv(
    csv_path: Path,
) -> tuple[list[str], list[dict]]:
    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing_fields = ORIGINAL_REQUIRED_FIELDS - set(fieldnames)

    if missing_fields:
        raise RuntimeError(
            "images.csv is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} rows in images.csv, "
            f"but found {len(rows)}."
        )

    return fieldnames, rows


def validate_rows(rows: list[dict]) -> None:
    image_ids = []
    file_paths = []

    for row_number, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        file_path = row["file_path"].strip()

        if not image_id:
            raise RuntimeError(
                f"Empty image_id in CSV row {row_number}."
            )

        if not file_path:
            raise RuntimeError(
                f"Empty file_path for {image_id}."
            )

        try:
            pixel_width = int(row["pixel_width"])
            pixel_height = int(row["pixel_height"])
            occurrence_count = int(row["occurrence_count"])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric value for {image_id}."
            ) from error

        if pixel_width <= 0 or pixel_height <= 0:
            raise RuntimeError(
                f"Invalid dimensions for {image_id}."
            )

        if occurrence_count <= 0:
            raise RuntimeError(
                f"Invalid occurrence_count for {image_id}."
            )

        image_ids.append(image_id)
        file_paths.append(file_path)

    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(
            "images.csv contains duplicate image_id values."
        )

    if len(file_paths) != len(set(file_paths)):
        raise RuntimeError(
            "images.csv contains duplicate file_path values."
        )


def resolve_image_path(
    project_root: Path,
    recorded_path: str,
) -> Path:
    normalized_path = recorded_path.replace("\\", "/")
    image_path = project_root / normalized_path

    return image_path.resolve()


def validate_image_files(
    rows: list[dict],
    project_root: Path,
) -> None:
    missing_files = []
    unreadable_files = []
    dimension_mismatches = []

    for row in rows:
        image_id = row["image_id"].strip()

        image_path = resolve_image_path(
            project_root=project_root,
            recorded_path=row["file_path"],
        )

        if not image_path.is_file():
            missing_files.append(
                (image_id, str(image_path))
            )
            continue

        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
                image.verify()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):
            unreadable_files.append(image_id)
            continue

        expected_width = int(row["pixel_width"])
        expected_height = int(row["pixel_height"])

        if (
            actual_width != expected_width
            or actual_height != expected_height
        ):
            dimension_mismatches.append(
                (
                    image_id,
                    expected_width,
                    expected_height,
                    actual_width,
                    actual_height,
                )
            )

    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} image file(s) are missing. "
            f"First missing files: {missing_files[:5]}"
        )

    if unreadable_files:
        raise RuntimeError(
            f"{len(unreadable_files)} image file(s) cannot be "
            f"opened. First files: {unreadable_files[:10]}"
        )

    if dimension_mismatches:
        raise RuntimeError(
            f"{len(dimension_mismatches)} image file(s) have "
            "dimensions different from images.csv. "
            f"First mismatches: {dimension_mismatches[:5]}"
        )


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def load_clip(
    device: str,
) -> tuple[CLIPModel, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return model, processor


def build_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> tuple[list[str], torch.Tensor]:
    class_names = list(CLASS_PROMPTS.keys())
    class_feature_list = []

    with torch.inference_mode():
        for class_name in class_names:
            prompts = CLASS_PROMPTS[class_name]

            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs[
                "attention_mask"
            ].to(device)

            text_outputs = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            text_features = text_outputs.pooler_output

            text_features = text_features / (
                text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-12
            )

            class_feature = text_features.mean(
                dim=0,
                keepdim=True,
            )

            class_feature = class_feature / (
                class_feature.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-12
            )

            class_feature_list.append(class_feature)

    class_features = torch.cat(
        class_feature_list,
        dim=0,
    )

    return class_names, class_features


def load_image_for_clip(
    image_path: Path,
) -> Image.Image:
    with Image.open(image_path) as image:
        converted_image = image.convert("RGB")

    return converted_image


def structural_override(
    predicted_type: str,
    confidence: float,
    margin: float,
    pixel_width: int,
    pixel_height: int,
) -> tuple[str, str]:
    pixel_area = pixel_width * pixel_height

    very_small = (
        pixel_width <= 24
        or pixel_height <= 12
        or pixel_area <= 500
    )

    small_fragment = (
        pixel_width <= 45
        or pixel_height <= 20
        or pixel_area <= 1200
    )

    if very_small:
        return (
            "fragment_or_noise",
            "clip_with_structural_rules",
        )

    if (
        small_fragment
        and predicted_type
        in {
            "document_element",
            "decorative_or_logo",
            "fragment_or_noise",
        }
    ):
        return (
            "fragment_or_noise",
            "clip_with_structural_rules",
        )

    if confidence < 0.22 or margin < 0.015:
        return (
            "uncertain",
            "clip_with_structural_rules",
        )

    return predicted_type, "clip_with_structural_rules"


def determine_review_status(
    predicted_type: str,
    confidence: float,
    margin: float,
) -> str:
    if predicted_type == "uncertain":
        return "needs_review"

    if confidence < 0.35:
        return "needs_review"

    if margin < 0.05:
        return "needs_review"

    return "predicted_needs_confirmation"


def classify_image(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    class_names: list[str],
    class_features: torch.Tensor,
    device: str,
    pixel_width: int,
    pixel_height: int,
) -> dict:
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        image_outputs = model.get_image_features(
            pixel_values=pixel_values
        )

        image_features = image_outputs.pooler_output

        image_features = image_features / (
            image_features.norm(
                dim=-1,
                keepdim=True,
            )
            + 1e-12
        )

        similarities = (
            100.0
            * image_features
            @ class_features.T
        )

        probabilities = similarities.softmax(
            dim=-1
        )[0]

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
    )

    best_index = int(sorted_indices[0].item())
    second_index = int(sorted_indices[1].item())

    raw_type = class_names[best_index]
    confidence = float(
        sorted_probabilities[0].item()
    )
    second_probability = float(
        sorted_probabilities[1].item()
    )
    margin = confidence - second_probability

    predicted_type, method = structural_override(
        predicted_type=raw_type,
        confidence=confidence,
        margin=margin,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    review_status = determine_review_status(
        predicted_type=predicted_type,
        confidence=confidence,
        margin=margin,
    )

    return {
        "predicted_type": predicted_type,
        "classification_confidence": (
            f"{confidence:.6f}"
        ),
        "classification_margin": f"{margin:.6f}",
        "classification_method": (
            f"{method}:{MODEL_NAME}"
        ),
        "review_status": review_status,
        "final_type": "",
        "content_relevance": "undetermined",
    }


def classify_all_images(
    rows: list[dict],
    project_root: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    class_names: list[str],
    class_features: torch.Tensor,
    device: str,
) -> None:
    total_images = len(rows)

    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(
            project_root=project_root,
            recorded_path=row["file_path"],
        )

        image = load_image_for_clip(image_path)

        result = classify_image(
            image=image,
            model=model,
            processor=processor,
            class_names=class_names,
            class_features=class_features,
            device=device,
            pixel_width=int(row["pixel_width"]),
            pixel_height=int(row["pixel_height"]),
        )

        row.update(result)

        if (
            index == 1
            or index % 25 == 0
            or index == total_images
        ):
            print(
                f"[CLASSIFY] {index}/{total_images}"
            )


def build_output_fields(
    original_fields: list[str],
) -> list[str]:
    output_fields = [
        field
        for field in original_fields
        if field not in CLASSIFICATION_FIELDS
    ]

    output_fields.extend(CLASSIFICATION_FIELDS)

    return output_fields


def safely_update_csv(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    temporary_path = csv_path.with_name(
        f"{csv_path.name}.writing"
    )

    if temporary_path.exists():
        raise FileExistsError(
            "An unfinished write file already exists: "
            f"{temporary_path}"
        )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

            csv_file.flush()
            os.fsync(csv_file.fileno())

        with temporary_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            verification_reader = csv.DictReader(csv_file)
            verification_rows = list(
                verification_reader
            )

        if len(verification_rows) != EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                "The updated CSV failed row-count validation."
            )

        for row in verification_rows:
            if not row["predicted_type"].strip():
                raise RuntimeError(
                    "The updated CSV contains an empty "
                    "predicted_type."
                )

            if row["final_type"].strip():
                raise RuntimeError(
                    "final_type must remain empty before "
                    "human review."
                )

            if (
                row["content_relevance"].strip()
                != "undetermined"
            ):
                raise RuntimeError(
                    "content_relevance must remain "
                    "undetermined before review."
                )

        temporary_path.replace(csv_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def summarize_results(rows: list[dict]) -> None:
    type_counts = {}
    review_counts = {}

    for row in rows:
        predicted_type = row["predicted_type"]
        review_status = row["review_status"]

        type_counts[predicted_type] = (
            type_counts.get(predicted_type, 0) + 1
        )

        review_counts[review_status] = (
            review_counts.get(review_status, 0) + 1
        )

    print("\nPredicted image types:")

    for image_type in sorted(type_counts):
        print(
            f"  {image_type}: "
            f"{type_counts[image_type]}"
        )

    print("\nReview status:")

    for status in sorted(review_counts):
        print(
            f"  {status}: {review_counts[status]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images-csv",
        default="data/graph_v2/images.csv",
        help="Path to images.csv.",
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the project.",
    )

    args = parser.parse_args()

    configure_csv_field_limit()

    project_root = Path(
        args.project_root
    ).resolve()

    csv_path = Path(args.images_csv)

    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    csv_path = csv_path.resolve()

    if not project_root.is_dir():
        raise NotADirectoryError(
            f"Project root not found: {project_root}"
        )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"images.csv not found: {csv_path}"
        )

    print("[1/6] Reading images.csv...")

    original_fields, rows = read_images_csv(
        csv_path
    )

    validate_rows(rows)

    print(
        f"[OK] Validated {len(rows)} CSV rows"
    )

    print("[2/6] Validating image files...")

    validate_image_files(
        rows=rows,
        project_root=project_root,
    )

    print(
        f"[OK] Validated {len(rows)} image files"
    )

    device = select_device()

    print(
        f"[3/6] Loading CLIP model on {device}..."
    )

    model, processor = load_clip(device)

    print("[OK] CLIP model loaded")

    print("[4/6] Building text features...")

    class_names, class_features = (
        build_text_features(
            model=model,
            processor=processor,
            device=device,
        )
    )

    print(
        f"[OK] Prepared {len(class_names)} classes"
    )

    print("[5/6] Classifying raster images...")

    classify_all_images(
        rows=rows,
        project_root=project_root,
        model=model,
        processor=processor,
        class_names=class_names,
        class_features=class_features,
        device=device,
    )

    output_fields = build_output_fields(
        original_fields
    )

    print("[6/6] Updating images.csv...")

    safely_update_csv(
        csv_path=csv_path,
        fieldnames=output_fields,
        rows=rows,
    )

    print("[OK] images.csv updated safely")
    print("[OK] No image file was removed")
    print("[OK] final_type remains empty")
    print(
        "[OK] content_relevance remains undetermined"
    )

    summarize_results(rows)


if __name__ == "__main__":
    main()