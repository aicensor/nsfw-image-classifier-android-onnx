// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier;

import ai.onnxruntime.*;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.SystemClock;
import android.util.Log;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.function.Consumer;

public class ORTAnalyzer implements ImageAnalysis.Analyzer {
    private OrtSession ortSession;
    private Consumer<Result> callBack;

    public ORTAnalyzer(OrtSession ortSession, Consumer<Result> callBack) {
        this.ortSession = ortSession;
        this.callBack = callBack;
    }

    // Get index of top 3 values
    // This is for demo purpose only, there are more efficient algorithms for topK problems
    private List<Integer> getTop3(float[] labelVals) {
        List<Integer> indices = new ArrayList<>();
        for (int k = 0; k < 3; k++) {
            float max = 0.0f;
            int idx = 0;
            for (int i = 0; i < labelVals.length; i++) {
                float labelVal = labelVals[i];
                if (labelVal > max && !indices.contains(i)) {
                    max = labelVal;
                    idx = i;
                }
            }
            indices.add(idx);
        }
        return indices;
    }

    // Calculate the SoftMax for the input array
    private float[] softMax(float[] modelResult) {
        float[] labelVals = modelResult.clone();
        float max = 0f;
        for (float val : labelVals) {
            if (val > max) max = val;
        }
        float sum = 0.0f;

        // Get the reduced sum
        for (int i = 0; i < labelVals.length; i++) {
            labelVals[i] = (float) Math.exp(labelVals[i] - max);
            sum += labelVals[i];
        }

        if (sum != 0.0f) {
            for (int i = 0; i < labelVals.length; i++) {
                labelVals[i] /= sum;
            }
        }

        return labelVals;
    }

    // Convert ImageProxy to Bitmap with sampling for high-resolution images
    private Bitmap imageProxyToBitmapWithSampling(ImageProxy image, int sampleSize) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ImageProxy.PlaneProxy plane = planes[0];
        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();
        
        // Calculate sampled dimensions
        int sampledWidth = originalWidth / sampleSize;
        int sampledHeight = originalHeight / sampleSize;
        
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        
        // Create bitmap with sampled dimensions
        Bitmap bitmap = Bitmap.createBitmap(sampledWidth, sampledHeight, Bitmap.Config.ARGB_8888);
        
        // Get the buffer
        java.nio.ByteBuffer buffer = plane.getBuffer();
        buffer.rewind();
        
        // Create a temporary buffer for the pixel data
        int[] pixels = new int[sampledWidth * sampledHeight];
        
        // Read pixel data from buffer with sampling
        for (int y = 0; y < sampledHeight; y++) {
            for (int x = 0; x < sampledWidth; x++) {
                int pixelIndex = y * sampledWidth + x;
                int originalY = y * sampleSize;
                int originalX = x * sampleSize;
                int bufferIndex = originalY * rowStride + originalX * pixelStride;
                
                if (bufferIndex < buffer.limit()) {
                    // For YUV format, we only use the Y channel for grayscale
                    int yValue = buffer.get(bufferIndex) & 0xFF;
                    pixels[pixelIndex] = (0xFF << 24) | (yValue << 16) | (yValue << 8) | yValue;
                }
            }
        }
        
        bitmap.setPixels(pixels, 0, sampledWidth, 0, 0, sampledWidth, sampledHeight);
        return bitmap;
    }

    // Convert ImageProxy to Bitmap (original method for backward compatibility)
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        return imageProxyToBitmapWithSampling(image, 1);
    }

    // Rotate the image of the input bitmap
    public static Bitmap rotate(Bitmap bitmap, float degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    @Override
    public void analyze(ImageProxy image) {
        // For high-resolution images, we need to optimize memory usage
        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();
        
        // Calculate optimal sampling size to avoid memory issues
        // For very large images, we'll sample down before creating the full bitmap
        int maxDimension = Math.max(originalWidth, originalHeight);
        int sampleSize = 1;
        
        if (maxDimension > 4096) {
            // For very large images (>4K), sample down first
            sampleSize = maxDimension / 2048;
        } else if (maxDimension > 2048) {
            sampleSize = maxDimension / 1024;
        }
        
        // Convert the input image to bitmap with sampling
        Bitmap imgBitmap = imageProxyToBitmapWithSampling(image, sampleSize);
        
        // Resize to 320x320 for NudeNet model input
        Bitmap rawBitmap = Bitmap.createScaledBitmap(imgBitmap, 320, 320, false);
        Bitmap bitmap = rotate(rawBitmap, image.getImageInfo().getRotationDegrees());
        
        // Free the intermediate bitmap to save memory
        if (imgBitmap != rawBitmap) {
            imgBitmap.recycle();
        }

        if (bitmap != null) {
            Result result = new Result();

            try {
                FloatBuffer imgData = ImageUtil.preProcessForNudeNet(bitmap);
                String inputName = ortSession.getInputNames().iterator().next();
                long[] shape = {1, 3, 320, 320}; // [batch, channels, height, width] for NudeNet model
                OrtEnvironment env = OrtEnvironment.getEnvironment();
                
                try (OnnxTensor tensor = OnnxTensor.createTensor(env, imgData, shape)) {
                    long startTime = SystemClock.uptimeMillis();
                    try (OrtSession.Result output = ortSession.run(Collections.singletonMap(inputName, tensor))) {
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime;
                    
                    // NudeNet outputs detection results as 3D array [batch, features, detections]
                    // Need to transpose to get [detections, features] format like the original script
                    @SuppressWarnings("unchecked")
                    float[][][] rawOutput = (float[][][]) output.get(0).getValue();
                    float[][] outputArray = rawOutput[0];
                    
                    // Log the output structure for debugging
                    Log.d("NudeNet", "Raw output shape: [" + outputArray.length + "] features");
                    if (outputArray.length > 0) {
                        Log.d("NudeNet", "First feature size: " + outputArray[0].length);
                    }
                    
                    // Transpose the output to get [detections, features] format
                    // Each detection has 22 values: [x, y, w, h, class_scores...]
                    int numDetections = outputArray.length > 0 ? outputArray[0].length : 0;
                    int numFeatures = outputArray.length;
                    Log.d("NudeNet", "Transposed: " + numDetections + " detections, " + numFeatures + " features");
                    
                    // Find the detection with highest confidence
                    float maxScore = 0f;
                    int maxDetectionIndex = -1;
                    List<DetectionResult> detections = new ArrayList<>();
                    
                    // Process each detection
                    for (int i = 0; i < numDetections; i++) {
                        // Extract detection values: [x, y, w, h, class_scores...]
                        float x = outputArray[0][i];  // x coordinate
                        float y = outputArray[1][i];  // y coordinate  
                        float w = outputArray[2][i];  // width
                        float h = outputArray[3][i];  // height
                        
                        // Extract class scores (features 4-21)
                        float[] classScores = new float[18];
                        for (int j = 0; j < 18; j++) {
                            if (4 + j < numFeatures) {
                                classScores[j] = outputArray[4 + j][i];
                            }
                        }
                        
                        float maxClassScore = 0f;
                        for (float score : classScores) {
                            if (score > maxClassScore) maxClassScore = score;
                        }
                        int classIndex = -1;
                        for (int j = 0; j < classScores.length; j++) {
                            if (classScores[j] == maxClassScore) {
                                classIndex = j;
                                break;
                            }
                        }
                        
                        Log.d("NudeNet", "Detection " + i + ": x=" + x + ", y=" + y + ", w=" + w + ", h=" + h + ", maxScore=" + maxClassScore + ", class=" + classIndex);
                        
                        if (maxClassScore >= 0.2f) { // NudeNet threshold is 0.2
                            detections.add(new DetectionResult(
                                x, y, w, h, maxClassScore, classIndex
                            ));
                            
                            if (maxClassScore > maxScore) {
                                maxScore = maxClassScore;
                                maxDetectionIndex = i;
                            }
                        }
                    }
                    
                    Log.d("NudeNet", "Found " + detections.size() + " detections with confidence >= 0.2");
                    Log.d("NudeNet", "Max confidence: " + maxScore + " at detection " + maxDetectionIndex);
                    
                    // Store detection results for UI display
                    result.detections = detections;
                    
                    // For compatibility, set some basic values
                    List<Integer> detectedIndices = new ArrayList<>();
                    List<Float> detectedScores = new ArrayList<>();
                    for (DetectionResult detection : detections) {
                        detectedIndices.add(detection.classIndex);
                        detectedScores.add(detection.confidence);
                    }
                    result.detectedIndices = detectedIndices;
                    result.detectedScore = detectedScores;
                    }
                }
            } catch (Exception e) {
                Log.e("NudeNet", "Error processing image", e);
            } finally {
                // Always clean up bitmaps to prevent memory leaks
                if (bitmap != null) {
                    bitmap.recycle();
                }
                if (rawBitmap != null && rawBitmap != bitmap) {
                    rawBitmap.recycle();
                }
            }
            
            callBack.accept(result);
        }

        image.close();
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    public void cleanup() {
        if (ortSession != null) {
            try {
                ortSession.close();
            } catch (OrtException e) {
                Log.e("ORTAnalyzer", "Error closing ORT session", e);
            }
        }
    }
}
