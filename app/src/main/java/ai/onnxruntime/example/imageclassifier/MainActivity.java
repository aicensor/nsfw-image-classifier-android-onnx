// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier;

import ai.onnxruntime.*;
import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainBinding;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private ExecutorService backgroundExecutor;
    private List<String> labelData;
    private OrtEnvironment ortEnv;
    private ImageCapture imageCapture;
    private ImageAnalysis imageAnalysis;

    private static final String TAG = "ORTImageClassifier";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        
        backgroundExecutor = Executors.newSingleThreadExecutor();
        labelData = readLabels();
        ortEnv = OrtEnvironment.getEnvironment();
        
        // Request Camera permission
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            );
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                    // Preview
                    Preview preview = new Preview.Builder()
                            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                            .build();
                    preview.setSurfaceProvider(binding.viewFinder.getSurfaceProvider());

                    imageCapture = new ImageCapture.Builder()
                            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                            .build();

                    CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                    imageAnalysis = new ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build();

                    try {
                        cameraProvider.unbindAll();

                        cameraProvider.bindToLifecycle(
                                MainActivity.this, cameraSelector, preview, imageCapture, imageAnalysis
                        );
                    } catch (Exception exc) {
                        Log.e(TAG, "Use case binding failed", exc);
                    }

                    setORTAnalyzer();
                } catch (Exception e) {
                    Log.e(TAG, "Camera provider failed", e);
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(getBaseContext(), permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
        if (ortEnv != null) {
            ortEnv.close();
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(
                        this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT
                ).show();
                finish();
            }
        }
    }

    private void updateUI(Result result) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Update bounding box overlay with all detections
                binding.boundingBoxOverlay.updateDetections(result.detections);
                
                // Show detection count and summary
                int detectionCount = result.detections.size();
                if (detectionCount > 0) {
                    // Sort detections by confidence (descending)
                    List<DetectionResult> sortedDetections = new ArrayList<>(result.detections);
                    sortedDetections.sort((a, b) -> Float.compare(b.confidence, a.confidence));
                    DetectionResult topDetection = sortedDetections.get(0);
                    
                    // Show detection summary
                    binding.detectedItem1.setText("Detections: " + detectionCount);
                    binding.detectedItemValue1.setText("Top: " + (int)(topDetection.confidence * 100) + "%");
                    
                    // Color code based on highest confidence
                    int detectionColor = topDetection.confidence > 0.5f ? 
                        android.graphics.Color.RED : android.graphics.Color.BLUE;
                    binding.detectedItem1.setTextColor(detectionColor);
                    binding.detectedItemValue1.setTextColor(detectionColor);
                    
                    // Update progress bar with top detection confidence
                    binding.percentMeter.setProgress((int)(topDetection.confidence * 100));
                    
                    // Show class breakdown
                    // Group by class index and count occurrences
                    int[] classCounts = new int[18]; // NudeNet has 18 classes
                    for (DetectionResult detection : result.detections) {
                        if (detection.classIndex < classCounts.length) {
                            classCounts[detection.classIndex]++;
                        }
                    }
                    
                    // Find top two classes
                    int maxCount = 0;
                    int maxClass = -1;
                    int secondCount = 0;
                    int secondClass = -1;
                    
                    for (int i = 0; i < classCounts.length; i++) {
                        if (classCounts[i] > maxCount) {
                            secondCount = maxCount;
                            secondClass = maxClass;
                            maxCount = classCounts[i];
                            maxClass = i;
                        } else if (classCounts[i] > secondCount) {
                            secondCount = classCounts[i];
                            secondClass = i;
                        }
                    }
                    
                    if (maxClass >= 0) {
                        String classLabel = maxClass < labelData.size() ? 
                            labelData.get(maxClass) : "UNKNOWN_" + maxClass;
                        binding.detectedItem2.setText("Most detected: " + classLabel);
                        binding.detectedItemValue2.setText("Count: " + maxCount);
                    } else {
                        binding.detectedItem2.setText("");
                        binding.detectedItemValue2.setText("");
                    }
                    
                    if (secondClass >= 0) {
                        String classLabel = secondClass < labelData.size() ? 
                            labelData.get(secondClass) : "UNKNOWN_" + secondClass;
                        binding.detectedItem3.setText("Second: " + classLabel);
                        binding.detectedItemValue3.setText("Count: " + secondCount);
                    } else {
                        binding.detectedItem3.setText("");
                        binding.detectedItemValue3.setText("");
                    }
                } else {
                    binding.detectedItem1.setText("No detections");
                    binding.detectedItemValue1.setText("");
                    binding.detectedItem1.setTextColor(android.graphics.Color.GRAY);
                    binding.detectedItemValue1.setTextColor(android.graphics.Color.GRAY);
                    binding.percentMeter.setProgress(0);
                    binding.detectedItem2.setText("");
                    binding.detectedItemValue2.setText("");
                    binding.detectedItem3.setText("");
                    binding.detectedItemValue3.setText("");
                }

                binding.inferenceTimeValue.setText(result.processTimeMs + "ms");
            }
        });
    }

    // Read NSFW classification labels
    private List<String> readLabels() {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(getResources().openRawResource(R.raw.nsfw_classes)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading labels", e);
        }
        return labels;
    }

    // Read NudeNet 320n model into a ByteArray, run in background
    private byte[] readModel() {
        try {
            java.io.InputStream inputStream = getResources().openRawResource(R.raw.nudenet_320n);
            java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[16384];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            inputStream.close();
            return buffer.toByteArray();
        } catch (IOException e) {
            Log.e(TAG, "Error reading model", e);
            return new byte[0];
        }
    }

    // Create a new ORT session in background
    private OrtSession createOrtSession() {
        try {
            return ortEnv.createSession(readModel());
        } catch (Exception e) {
            Log.e(TAG, "Error creating ORT session", e);
            return null;
        }
    }

    // Create a new ORT session and then change the ImageAnalysis.Analyzer
    // This part is done in background to avoid blocking the UI
    private void setORTAnalyzer() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                if (imageAnalysis != null) {
                    imageAnalysis.clearAnalyzer();
                    
                    // Create analyzer with memory-optimized settings
                    ORTAnalyzer analyzer = new ORTAnalyzer(createOrtSession(), MainActivity.this::updateUI);
                    
                    // Set analyzer with optimized backpressure strategy
                    imageAnalysis.setAnalyzer(
                            backgroundExecutor,
                            analyzer
                    );
                }
            }
        }).start();
    }
}
