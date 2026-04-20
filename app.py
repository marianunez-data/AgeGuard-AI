import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_PATH = "models/onnx/ageguard_v1.onnx"
LEGAL_AGE = 21
ALERT_THRESHOLD = 25

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def predict_age(image):
    if image is None:
        return "Please upload a face image"
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    pred = session.run(None, {"image": img})[0]
    age = int(round(float(pred.flatten()[0])))
    age = max(0, min(100, age))

    if age < LEGAL_AGE:
        emoji = "🔴"
        status = "BLOCK"
        action = "Sale denied — predicted minor"
        color = "#FF4444"
    elif age < ALERT_THRESHOLD:
        emoji = "🟡"
        status = "VERIFY"
        action = "Request ID — age near legal threshold"
        color = "#FFB300"
    else:
        emoji = "🟢"
        status = "APPROVED"
        action = "Sale approved — predicted adult"
        color = "#00C853"

    result = f"""
<div style="text-align:center; padding:20px;">
    <div style="font-size:48px; font-weight:bold; color:{color}; margin-bottom:10px;">
        {emoji} {age} years old
    </div>
    <div style="font-size:24px; font-weight:bold; color:{color}; margin-bottom:20px;">
        {status}
    </div>
    <div style="background:{color}22; border:2px solid {color}; border-radius:12px; padding:16px; margin-bottom:20px;">
        <span style="font-size:18px;">{action}</span>
    </div>
    <div style="background:#1a1a2e; border-radius:8px; padding:12px; color:#aaa; font-size:13px;">
        ⚙️ Legal age: {LEGAL_AGE} | Alert threshold: {ALERT_THRESHOLD} | Model: EfficientNetV2-S | Inference: ONNX Runtime
    </div>
</div>
"""
    return result


with gr.Blocks(
    title="AgeGuard AI",
    css="""
    .gradio-container {max-width: 900px !important;}
    footer {display: none !important;}
""",
) as demo:
    gr.Markdown(
        """
    <div style="text-align:center; padding:20px 0;">
        <h1>🛡️ AgeGuard AI</h1>
        <h3 style="color:#888;">Automated Age Verification for Retail Compliance</h3>
        <p style="color:#666;">Upload a face image or use your webcam. The system estimates age and triggers the appropriate alert level for age-restricted sales.</p>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📷 Face Image", type="numpy", height=350)
            submit_btn = gr.Button("🔍 Verify Age", variant="primary", size="lg")
            clear_btn = gr.ClearButton([input_image], value="🗑️ Clear")

        with gr.Column(scale=1):
            output = gr.HTML(label="Result")

    submit_btn.click(fn=predict_age, inputs=input_image, outputs=output)

    gr.Markdown(
        """
    <div style="text-align:center; padding:15px; color:#666; font-size:12px; border-top:1px solid #333; margin-top:20px;">
        <strong>How it works:</strong> EfficientNetV2-S analyzes facial features (skin texture, bone structure, wrinkles) to estimate age.
        Trained on 7,446 face-cropped images · MAE: 5.02 years · Inference: ~17ms
        <br><br>
        🔴 <strong>BLOCK</strong> (&lt; 21): Deny sale | 🟡 <strong>VERIFY</strong> (21-24): Request ID | 🟢 <strong>APPROVED</strong> (&ge; 25): Approve sale
    </div>
    """
    )

if __name__ == "__main__":
    demo.launch()
