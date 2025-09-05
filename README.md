# NSFW Image Classifier Android ONNX

## Overview
This is an Android application for NSFW (Not Safe For Work) image classification using [ONNX Runtime](https://github.com/microsoft/onnxruntime). The demo app uses image classification which is able to continuously classify images from the device's camera in real-time and displays the most probable inference results on the screen.

## Source Code Attribution
This source code is based on:
- [onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples) - Microsoft's ONNX Runtime inference examples
- [nudenet v3.4 320n onnx](https://github.com/notAI-tech/NudeNet) - NudeNet model for NSFW detection

This example is also loosely based on [Google CodeLabs - Getting Started with CameraX](https://codelabs.developers.google.com/codelabs/camerax-getting-started)

### Model
This application uses the [NudeNet v3.4 320n ONNX model](https://github.com/notAI-tech/NudeNet) for NSFW content detection. The model has been converted to ONNX format for efficient inference on Android devices.

## Requirements
- Android Studio Electric Eel 2022.1.1+ (installed on Mac/Windows/Linux)
- Android device with a camera in [developer mode](https://developer.android.com/studio/debug/dev-options) with USB debugging enabled

## Build And Run

### Step 1. Clone the ONNX Runtime Mobile examples source code and download required model files
Clone this GitHub repository to your computer to get the sample application.

Run `mobile/examples/image_classification/android/prepare_models.py` to download and prepare the labels file and model files in the sample application resource directory.

```bash
cd mobile/examples/image_classification/android  # cd to this directory
python -m pip install -r ./prepare_models.requirements.txt
python ./prepare_models.py --output_dir ./app/src/main/res/raw
```

Then open the sample application in Android Studio. To do this, open Android Studio and select `Open an existing project`, browse folders and open the folder `mobile/examples/image_classification/android/`.

<img width=60% src="images/screenshot_1.png"/>

### Step 2. Build the sample application in Android Studio

Select `Build -> Make Project` in the top toolbar in Android Studio and check the projects has built successfully.

<img width=60% src="images/screenshot_3.png" alt="App Screenshot"/>

<img width=60% src="images/screenshot_4.png" alt="App Screenshot"/>

### Step 3. Connect your Android Device and run the app

Connect your Android Device to the computer and select your device in the top-down device bar.

<img width=60% src="images/screenshot_5.png" alt="App Screenshot"/>

<img width=60% src="images/screenshot_6.png" alt="App Screenshot"/>

Then Select `Run -> Run app` and this will prompt the app to be installed on your device.

Now you can test and try by opening the app `ort_image_classifier` on your device. The app may request your permission for using the camera.


#
Here's an example screenshot of the app.

<img width=20% src="images/screenshot_2.jpg" alt="App Screenshot" />



