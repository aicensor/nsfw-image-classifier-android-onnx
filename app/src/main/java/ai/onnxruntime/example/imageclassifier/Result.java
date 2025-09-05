package ai.onnxruntime.example.imageclassifier;

import java.util.ArrayList;
import java.util.List;

public class Result {
    public List<Integer> detectedIndices = new ArrayList<>();
    public List<Float> detectedScore = new ArrayList<>();
    public long processTimeMs = 0;
    public List<DetectionResult> detections = new ArrayList<>();
}
