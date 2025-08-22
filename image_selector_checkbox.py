import argparse
import json
import os
import csv
from PIL import Image
from io import BytesIO
import base64
import random
from shiny import App, ui, reactive, render

# Parse arguments
parser = argparse.ArgumentParser(description="Shiny app for image selection.")
parser.add_argument("target_dirs_json", help="Path to the JSON file containing target directories")
args = parser.parse_args()
sample_size = 18
im_size = 300

# Load available directories
with open(args.target_dirs_json, "r") as f:
    available_directories = json.load(f)

# Ensure log file exists
log_file = "verified_robust_images.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["basename", "src_dir"])
        writer.writeheader()

# Function to read directories already processed from the log file
def get_processed_dirs():
    processed_dirs = set()
    with open(log_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            processed_dirs.add(row["src_dir"])
    return processed_dirs

# Filter directories based on the log file
processed_dirs = get_processed_dirs()
filtered_dirs = {k: v for k, v in available_directories.items() if v not in processed_dirs}

# UI layout
app_ui = ui.page_fluid(
    ui.output_ui("folder_select"),
    ui.input_action_button("resample", "Resample Images"),
    ui.input_action_button("log_selected", "Log Selected Images"),
    ui.output_ui("image_gallery"),
    ui.input_action_button("delete_folder", "Discard folder"),
)

# Server logic
def server(input, output, session):
    # Reactive variables
    filtered_dirs_reactive = reactive.Value(filtered_dirs)
    image_data = reactive.Value([])

    # Generate thumbnails and manage image data
    def generate_image_data(directory, num_samples=sample_size):
        image_files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
        sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
        image_info = []
        for filename in sampled_files:
            img_path = os.path.join(directory, filename)
            try:
                with Image.open(img_path) as img:
                    img.thumbnail((im_size, im_size))
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
                    image_info.append({"src": f"data:image/jpeg;base64,{encoded_img}", "name": filename, "selected": False})
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
        return image_info

    # Folder selection UI
    @output
    @render.ui
    def folder_select():
        dirs = filtered_dirs_reactive.get()
        return ui.input_select("folder", "Select folder:", choices=dirs, selected=None)

    # Image gallery UI
    @output
    @render.ui
    def image_gallery():
        images = image_data.get()
        gallery_items = [
            ui.tags.div(  # Wrap each checkbox and image in a div
                ui.input_checkbox(f"cb_{i}", "", value=img["selected"]),
                ui.tags.img(src=img["src"], height=im_size),
                style="margin-right: 2px; display: inline-block;"  # Adjust style as needed
            )
            for i, img in enumerate(images)
        ]
        return ui.tags.div(*gallery_items, style="display: flex; flex-wrap: wrap; gap: 10px;")

    # Update image data when folder is selected or resampled
    @reactive.Effect
    @reactive.event(input.folder, input.resample)
    def update_image_data():
        folder = input.folder()
        if folder:
            directory = filtered_dirs_reactive.get()[folder]
            image_data.set(generate_image_data(directory))

    # Update selection state based on checkbox input
    @reactive.Effect
    @reactive.event(input.log_selected)
    def update_selection():
        for i in range(len(image_data.get())):
            cb_id = f"cb_{i}"
            if input[cb_id].get():
                images = image_data.get()
                images[i]["selected"] = True
                image_data.set(images)

    # Logging selected images
    @reactive.Effect
    @reactive.event(input.log_selected)
    def log_selections():
        folder = input.folder()
        if folder:
            directory = filtered_dirs_reactive.get()[folder]  # Get the directory path from the selected folder key
            selected_images = [img for img in image_data.get() if img["selected"]]
            if selected_images:
                with open(log_file, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=["basename", "src_dir"])
                    for img in selected_images:
                        writer.writerow({"basename": img["name"], "src_dir": directory})
                print(f"Logged {len(selected_images)} images from {directory}")
                # Remove directory from filtered_dirs_reactive
                update_dirs = {k: v for k, v in filtered_dirs_reactive.get().items() if k != folder}
                filtered_dirs_reactive.set(update_dirs)
                
    @reactive.Effect
    @reactive.event(input.delete_folder)
    def delete_folder():
        folder = input.folder()
        update_dirs = {k: v for k, v in filtered_dirs_reactive.get().items() if k != folder}
        filtered_dirs_reactive.set(update_dirs)

# Create and run the app
app = App(app_ui, server)
if __name__ == "__main__":
    app.run()
