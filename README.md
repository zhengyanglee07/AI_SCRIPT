# Image and Video Processing Tool

## Overview

This project is a powerful tool for processing images and videos. It includes features for removing text from images and videos, translating text in images, and applying effects to faces. The tool leverages advanced image processing techniques and machine learning models to deliver high-quality results.

## Features

### 1. Text Removal from Images

- Remove all text from images using opencv inpainting techniques.
- Supports various image formats (JPEG, PNG, etc.).

### 2. Text Removal from Videos

- Automatically detect and remove text from video frames.
- Supports various video formats (MP4, AVI, etc.).

### 3. Translation for Images

- Translate text found in images to different languages.
- Utilizes Optical Character Recognition (OCR) to extract text.
- Supports multiple languages for translation.

### 4. Apply Effects to Face

- Apply various effects to faces in images videos (e.g., blurring, filters).

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Text Removal from Images

To remove text from an image, use the following command:

```bash
python remove_text_image.py --input <input_image_path> --output <output_image_path>
```

### Text Removal from Videos

To remove text from a video, use the following command:

```bash
python remove_text_video.py --input <input_video_path> --output <output_video_path>
```

### Translation for Images

To translate text in an image, use the following command:

```bash
python translate_image.py --input <input_image_path> --output <output_image_path> --language <target_language>
```

### Apply Effects to Face

To apply effects to faces in an image or video, use the following command:

```bash
python apply_face_effects.py --input <input_image_or_video_path> --output <output_image_or_video_path> --effect <effect_type>
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for image processing.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition.
- [Google Translate API](https://cloud.google.com/translate/docs) for translation services.
