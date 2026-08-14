import csv
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import pymupdf
from PIL import Image, ImageTk, UnidentifiedImageError


EXPECTED_IMAGE_COUNT = 753

FINAL_TYPES = [
    "microscopy",
    "clinical_or_laboratory",
    "diagram_or_chart",
    "table_or_form",
    "document_element",
    "decorative_or_logo",
    "fragment_or_noise",
    "uncertain",
]

RELEVANCE_OPTIONS = [
    "relevant",
    "not_relevant",
    "uncertain",
]

REQUIRED_FIELDS = {
    "image_id",
    "pixel_width",
    "pixel_height",
    "file_path",
    "first_pdf_page",
    "predicted_type",
    "classification_confidence",
    "classification_margin",
    "review_status",
    "final_type",
    "content_relevance",
}


def configure_csv_field_limit():
    maximum_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maximum_limit)
            return
        except OverflowError:
            maximum_limit //= 10


def resolve_image_path(project_root, recorded_path):
    normalized_path = recorded_path.replace("\\", "/")
    image_path = project_root / normalized_path

    return image_path.resolve()


def read_images_csv(csv_path):
    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing_fields = REQUIRED_FIELDS - set(fieldnames)

    if missing_fields:
        raise RuntimeError(
            "images.csv is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} rows, "
            f"but found {len(rows)}."
        )

    return fieldnames, rows


def validate_rows(rows, project_root):
    image_ids = []
    missing_files = []

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

        image_path = resolve_image_path(
            project_root,
            file_path,
        )

        if not image_path.is_file():
            missing_files.append(
                (image_id, str(image_path))
            )

        image_ids.append(image_id)

    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(
            "Duplicate image_id values found."
        )

    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} image files are missing. "
            f"First missing files: {missing_files[:5]}"
        )


def review_priority(row):
    review_status = row["review_status"].strip()
    predicted_type = row["predicted_type"].strip()

    if review_status == "needs_review":
        return 1

    if predicted_type == "fragment_or_noise":
        return 2

    if predicted_type in {
        "microscopy",
        "clinical_or_laboratory",
    }:
        return 3

    return 4


def build_review_queue(rows):
    queue = []

    for index, row in enumerate(rows):
        if row["review_status"].strip() == "human_reviewed":
            continue

        queue.append(index)

    queue.sort(
        key=lambda row_index: (
            review_priority(rows[row_index]),
            rows[row_index]["first_pdf_page"],
            rows[row_index]["image_id"],
        )
    )

    return queue


def safely_save_csv(csv_path, fieldnames, rows):
    temporary_path = csv_path.with_name(
        f"{csv_path.name}.review_writing"
    )

    if temporary_path.exists():
        raise FileExistsError(
            "An unfinished temporary file already exists:\n"
            f"{temporary_path}\n\n"
            "Close the program and inspect this file before "
            "continuing."
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
            verification_rows = list(verification_reader)

        if len(verification_rows) != EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                "Saved CSV failed row-count validation."
            )

        original_ids = [
            row["image_id"]
            for row in rows
        ]

        saved_ids = [
            row["image_id"]
            for row in verification_rows
        ]

        if saved_ids != original_ids:
            raise RuntimeError(
                "Saved CSV failed image_id validation."
            )

        temporary_path.replace(csv_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


class ImageReviewApp:
    def __init__(
        self,
        root,
        project_root,
        csv_path,
        fieldnames,
        rows,
    ):
        self.root = root
        self.project_root = project_root
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.rows = rows

        self.review_queue = build_review_queue(rows)
        self.queue_position = 0
        self.current_row_index = None
        self.current_photo = None

        self.final_type_var = tk.StringVar()
        self.relevance_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.root.title(
            "Raster Image Review"
        )
        self.root.geometry("1100x820")
        self.root.minsize(900, 700)

        self.build_interface()
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application,
        )

        if self.review_queue:
            self.show_current_image()
        else:
            self.show_completion_message()

    def build_interface(self):
        main_frame = ttk.Frame(
            self.root,
            padding=12,
        )
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 10),
        )

        self.progress_label = ttk.Label(
            progress_frame,
            text="",
            font=("Segoe UI", 11, "bold"),
        )
        self.progress_label.pack(
            side=tk.LEFT,
        )

        self.status_label = ttk.Label(
            progress_frame,
            textvariable=self.status_var,
            foreground="darkgreen",
        )
        self.status_label.pack(
            side=tk.RIGHT,
        )

        image_frame = ttk.LabelFrame(
            main_frame,
            text="Image",
            padding=10,
        )
        image_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 10),
        )

        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(
            image_frame,
            anchor=tk.CENTER,
        )
        self.image_label.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        details_frame = ttk.LabelFrame(
            main_frame,
            text="Image information",
            padding=12,
        )
        details_frame.grid(
            row=1,
            column=1,
            sticky="nsew",
        )

        details_frame.columnconfigure(
            1,
            weight=1,
        )

        self.detail_labels = {}

        detail_fields = [
            ("image_id", "Image ID"),
            ("first_pdf_page", "PDF page"),
            ("dimensions", "Dimensions"),
            ("predicted_type", "CLIP prediction"),
            ("classification_confidence", "Confidence"),
            ("classification_margin", "Margin"),
            ("review_status", "Current status"),
            ("file_path", "File path"),
        ]

        for row_number, (field, title) in enumerate(
            detail_fields
        ):
            title_label = ttk.Label(
                details_frame,
                text=f"{title}:",
                font=("Segoe UI", 10, "bold"),
            )
            title_label.grid(
                row=row_number,
                column=0,
                sticky="nw",
                padx=(0, 8),
                pady=4,
            )

            value_label = ttk.Label(
                details_frame,
                text="",
                wraplength=350,
                justify=tk.LEFT,
            )
            value_label.grid(
                row=row_number,
                column=1,
                sticky="nw",
                pady=4,
            )

            self.detail_labels[field] = value_label

        selection_start_row = len(detail_fields) + 1

        separator = ttk.Separator(
            details_frame,
            orient=tk.HORIZONTAL,
        )
        separator.grid(
            row=selection_start_row - 1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=12,
        )

        final_type_label = ttk.Label(
            details_frame,
            text="Final type:",
            font=("Segoe UI", 10, "bold"),
        )
        final_type_label.grid(
            row=selection_start_row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=6,
        )

        self.final_type_combo = ttk.Combobox(
            details_frame,
            textvariable=self.final_type_var,
            values=FINAL_TYPES,
            state="readonly",
            width=28,
        )
        self.final_type_combo.grid(
            row=selection_start_row,
            column=1,
            sticky="ew",
            pady=6,
        )

        relevance_label = ttk.Label(
            details_frame,
            text="Content relevance:",
            font=("Segoe UI", 10, "bold"),
        )
        relevance_label.grid(
            row=selection_start_row + 1,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=6,
        )

        self.relevance_combo = ttk.Combobox(
            details_frame,
            textvariable=self.relevance_var,
            values=RELEVANCE_OPTIONS,
            state="readonly",
            width=28,
        )
        self.relevance_combo.grid(
            row=selection_start_row + 1,
            column=1,
            sticky="ew",
            pady=6,
        )

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 0),
        )

        self.save_button = ttk.Button(
            button_frame,
            text="Save review and show next",
            command=self.save_and_next,
        )
        self.save_button.pack(
            side=tk.RIGHT,
        )

        self.use_prediction_button = ttk.Button(
            button_frame,
            text="Use CLIP prediction",
            command=self.use_prediction,
        )
        self.use_prediction_button.pack(
            side=tk.RIGHT,
            padx=(0, 10),
        )

        close_button = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_application,
        )
        close_button.pack(
            side=tk.LEFT,
        )

    def get_current_row(self):
        if self.current_row_index is None:
            return None

        return self.rows[self.current_row_index]

    def update_progress(self):
        reviewed_count = sum(
            1
            for row in self.rows
            if row["review_status"].strip()
            == "human_reviewed"
        )

        remaining_count = (
            EXPECTED_IMAGE_COUNT - reviewed_count
        )

        current_number = self.queue_position + 1
        queue_total = len(self.review_queue)

        self.progress_label.config(
            text=(
                f"Current: {current_number}/{queue_total}   |   "
                f"Human reviewed: {reviewed_count}/"
                f"{EXPECTED_IMAGE_COUNT}   |   "
                f"Remaining: {remaining_count}"
            )
        )

    def show_current_image(self):
        if self.queue_position >= len(
            self.review_queue
        ):
            self.show_completion_message()
            return

        self.current_row_index = self.review_queue[
            self.queue_position
        ]

        row = self.get_current_row()

        image_path = resolve_image_path(
            self.project_root,
            row["file_path"],
        )

        try:
            with Image.open(image_path) as image:
                display_image = image.convert("RGB")

            display_image.thumbnail(
                (650, 620),
                Image.Resampling.LANCZOS,
            )

            self.current_photo = ImageTk.PhotoImage(
                display_image
            )

            self.image_label.config(
                image=self.current_photo,
                text="",
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            messagebox.showerror(
                "Image error",
                f"Could not display:\n{image_path}\n\n{error}",
            )
            return

        confidence = self.format_probability(
            row["classification_confidence"]
        )

        margin = self.format_probability(
            row["classification_margin"]
        )

        self.detail_labels["image_id"].config(
            text=row["image_id"]
        )

        self.detail_labels["first_pdf_page"].config(
            text=row["first_pdf_page"]
        )

        self.detail_labels["dimensions"].config(
            text=(
                f'{row["pixel_width"]} × '
                f'{row["pixel_height"]} pixels'
            )
        )

        self.detail_labels["predicted_type"].config(
            text=row["predicted_type"]
        )

        self.detail_labels[
            "classification_confidence"
        ].config(text=confidence)

        self.detail_labels[
            "classification_margin"
        ].config(text=margin)

        self.detail_labels["review_status"].config(
            text=row["review_status"]
        )

        self.detail_labels["file_path"].config(
            text=row["file_path"]
        )

        existing_final_type = row[
            "final_type"
        ].strip()

        existing_relevance = row[
            "content_relevance"
        ].strip()

        if existing_final_type in FINAL_TYPES:
            self.final_type_var.set(
                existing_final_type
            )
        else:
            self.final_type_var.set("")

        if existing_relevance in RELEVANCE_OPTIONS:
            self.relevance_var.set(
                existing_relevance
            )
        else:
            self.relevance_var.set("")

        self.status_var.set("")
        self.update_progress()

    def format_probability(self, value):
        try:
            numeric_value = float(value)
            return (
                f"{numeric_value:.6f} "
                f"({numeric_value * 100:.2f}%)"
            )
        except (TypeError, ValueError):
            return value

    def use_prediction(self):
        row = self.get_current_row()

        if row is None:
            return

        predicted_type = row[
            "predicted_type"
        ].strip()

        if predicted_type in FINAL_TYPES:
            self.final_type_var.set(
                predicted_type
            )
        else:
            self.final_type_var.set(
                "uncertain"
            )

    def save_and_next(self):
        row = self.get_current_row()

        if row is None:
            return

        final_type = self.final_type_var.get().strip()
        relevance = self.relevance_var.get().strip()

        if final_type not in FINAL_TYPES:
            messagebox.showwarning(
                "Missing final type",
                "Select a valid final type.",
            )
            return

        if relevance not in RELEVANCE_OPTIONS:
            messagebox.showwarning(
                "Missing content relevance",
                "Select content relevance.",
            )
            return

        original_final_type = row[
            "final_type"
        ]

        original_relevance = row[
            "content_relevance"
        ]

        original_review_status = row[
            "review_status"
        ]

        row["final_type"] = final_type
        row["content_relevance"] = relevance
        row["review_status"] = "human_reviewed"

        try:
            safely_save_csv(
                self.csv_path,
                self.fieldnames,
                self.rows,
            )

        except Exception as error:
            row["final_type"] = original_final_type
            row[
                "content_relevance"
            ] = original_relevance
            row[
                "review_status"
            ] = original_review_status

            messagebox.showerror(
                "Save error",
                "The review was not saved.\n\n"
                f"{error}",
            )
            return

        self.status_var.set(
            f'Saved: {row["image_id"]}'
        )

        self.queue_position += 1

        if self.queue_position >= len(
            self.review_queue
        ):
            self.show_completion_message()
        else:
            self.show_current_image()

    def show_completion_message(self):
        self.current_row_index = None
        self.current_photo = None

        self.image_label.config(
            image="",
            text=(
                "All images have been human reviewed."
            ),
        )

        self.progress_label.config(
            text=(
                f"Human reviewed: "
                f"{EXPECTED_IMAGE_COUNT}/"
                f"{EXPECTED_IMAGE_COUNT}"
            )
        )

        self.status_var.set("Review complete")

        self.final_type_var.set("")
        self.relevance_var.set("")

        self.save_button.config(
            state=tk.DISABLED
        )

        self.use_prediction_button.config(
            state=tk.DISABLED
        )

        messagebox.showinfo(
            "Review complete",
            "All images have been human reviewed.",
        )

    def close_application(self):
        reviewed_count = sum(
            1
            for row in self.rows
            if row["review_status"].strip()
            == "human_reviewed"
        )

        should_close = messagebox.askyesno(
            "Close review tool",
            (
                f"{reviewed_count} of "
                f"{EXPECTED_IMAGE_COUNT} images have been "
                "reviewed.\n\n"
                "All confirmed reviews have already been "
                "saved.\n\n"
                "Close the program?"
            ),
        )

        if should_close:
            self.root.destroy()


def main():
    configure_csv_field_limit()

    project_root = Path(".").resolve()
    csv_path = (
        project_root
        / "data"
        / "graph_v2"
        / "images.csv"
    ).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"images.csv not found: {csv_path}"
        )

    fieldnames, rows = read_images_csv(
        csv_path
    )

    validate_rows(
        rows,
        project_root,
    )

    print(
        f"[OK] Validated {len(rows)} CSV rows "
        "and image files"
    )

    review_queue = build_review_queue(rows)

    reviewed_count = (
        EXPECTED_IMAGE_COUNT - len(review_queue)
    )

    print(
        f"[INFO] Already reviewed: {reviewed_count}"
    )
    print(
        f"[INFO] Waiting for review: "
        f"{len(review_queue)}"
    )

    root = tk.Tk()

    ImageReviewApp(
        root=root,
        project_root=project_root,
        csv_path=csv_path,
        fieldnames=fieldnames,
        rows=rows,
    )

    root.mainloop()


if __name__ == "__main__":
    main()