package ai.onnxruntime.example.imageclassifier;

import android.content.Context;
import android.graphics.*;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class BoundingBoxOverlay extends View {
    private List<DetectionResult> detections = new ArrayList<>();
    private Paint paint;
    private Paint textPaint;
    private Paint backgroundPaint;

    public BoundingBoxOverlay(Context context) {
        super(context);
        init();
    }

    public BoundingBoxOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public BoundingBoxOverlay(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4f);
        paint.setAntiAlias(true);
        
        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(32f);
        textPaint.setAntiAlias(true);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);
        
        backgroundPaint = new Paint();
        backgroundPaint.setColor(Color.BLACK);
        backgroundPaint.setAlpha(128);
    }

    public void updateDetections(List<DetectionResult> newDetections) {
        detections.clear();
        detections.addAll(newDetections);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        if (detections.isEmpty()) return;
        
        float viewWidth = getWidth();
        float viewHeight = getHeight();
        
        for (DetectionResult detection : detections) {
            // Convert normalized coordinates to screen coordinates
            float left = detection.x * viewWidth;
            float top = detection.y * viewHeight;
            float right = left + (detection.width * viewWidth);
            float bottom = top + (detection.height * viewHeight);
            
            // Set color based on confidence
            float confidence = detection.confidence;
            int color;
            if (confidence > 0.7f) {
                color = Color.RED;
            } else if (confidence > 0.5f) {
                color = Color.YELLOW;
            } else {
                color = Color.GREEN;
            }
            paint.setColor(color);
            
            // Draw bounding box
            canvas.drawRect(left, top, right, bottom, paint);
            
            // Draw label with confidence
            String label = getClassLabel(detection.classIndex);
            String confidenceText = label + ": " + (int)(confidence * 100) + "%";
            
            // Draw background for text
            Rect textBounds = new Rect();
            textPaint.getTextBounds(confidenceText, 0, confidenceText.length(), textBounds);
            float textX = left;
            float textY = top - 10f;
            RectF textBackground = new RectF(
                textX - 5f,
                textY - textBounds.height() - 5f,
                textX + textBounds.width() + 10f,
                textY + 5f
            );
            canvas.drawRect(textBackground, backgroundPaint);
            
            // Draw text
            canvas.drawText(confidenceText, textX, textY, textPaint);
        }
    }
    
    private String getClassLabel(int classIndex) {
        switch (classIndex) {
            case 0: return "FEMALE_GENITALIA_COVERED";
            case 1: return "FACE_FEMALE";
            case 2: return "BUTTOCKS_EXPOSED";
            case 3: return "FEMALE_BREAST_EXPOSED";
            case 4: return "FEMALE_GENITALIA_EXPOSED";
            case 5: return "MALE_BREAST_EXPOSED";
            case 6: return "ANUS_EXPOSED";
            case 7: return "FEET_EXPOSED";
            case 8: return "BELLY_EXPOSED";
            case 9: return "FEET_COVERED";
            case 10: return "ARMPITS_EXPOSED";
            case 11: return "ARMPITS_COVERED";
            case 12: return "FACE_MALE";
            case 13: return "BELLY_COVERED";
            case 14: return "MALE_GENITALIA_EXPOSED";
            case 15: return "BUTTOCKS_COVERED";
            case 16: return "FEMALE_BREAST_COVERED";
            case 17: return "MALE_GENITALIA_COVERED";
            default: return "UNKNOWN_" + classIndex;
        }
    }
}
