package ai.onnxruntime.example.imageclassifier;

public class DetectionResult {
    public final float x;
    public final float y;
    public final float width;
    public final float height;
    public final float confidence;
    public final int classIndex;

    public DetectionResult(float x, float y, float width, float height, float confidence, int classIndex) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.confidence = confidence;
        this.classIndex = classIndex;
    }
}
