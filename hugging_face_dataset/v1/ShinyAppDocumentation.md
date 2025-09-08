
# Image Selection and Copying Shiny App Documentation

## Overview

This Shiny application is designed to facilitate the selection and copying of images from various source directories to a specified destination directory. Users can preview, resample, and select images through a web interface, streamlining the process of handling and organizing image data.

## Environment Setup

Before running the application, ensure that your environment is prepared with the necessary dependencies. This guide assumes you have Python installed. Follow these steps:

1. **Install Required Libraries**: Install the Shiny library and other required packages using pip:

   ```sh
   pip install shiny Pillow
   ```

2. **Prepare the JSON File**: The app requires a JSON file specifying available source directories. Format the file as follows:

   ```json
   {
     "Directory Label 1": "/path/to/source/directory1",
     "Directory Label 2": "/path/to/source/directory2"
     // Add more directories as needed
   }
   ```

3. **Destination Directory**: Ensure you have a path in mind for the destination directory where selected images will be copied.

## Execution

To execute the app, run the Python script with arguments pointing to your JSON file and the desired destination directory:

```sh
python app.py /path/to/your/target_dirs.json /path/to/your/destination_directory
```

Replace `/path/to/your/target_dirs.json` with the path to your JSON file and `/path/to/your/destination_directory` with the path to your destination directory.

## Application Functionalities

The application consists of a user-friendly interface that provides the following functionalities:

1. **Directory Selection**: Users can select a source directory from the available options, which are dynamically filtered based on previous selections to avoid redundancy.

2. **Image Sampling and Resampling**: A random sample of images from the selected directory is displayed as thumbnails. Users can resample to view a different set of images.

3. **Image Preview and Selection**: Users can preview the sampled images and use a slider to select a specific image.

4. **Image Copying**: The selected image can be copied to the specified destination directory. The application logs each copied image to prevent duplicate processing in future sessions.

5. **Logging**: The application maintains a log of all copied images in a CSV file within the destination directory, recording the basename and source directory of each image.

## Notes

- The application handles image processing errors gracefully, ensuring stability even with corrupt or incompatible image files.
- Directories are dynamically updated in the selection menu based on the application's history, optimizing the user's workflow.

## Conclusion

This Shiny application streamlines the process of selecting and copying images from various sources, equipped with a user-friendly interface and efficient data handling capabilities. Ensure to follow the setup instructions carefully to prepare your environment for running the application.
