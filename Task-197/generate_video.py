import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob

# Import your existing model architecture
from model_ewc import CrowdContinualClassifier

VIRAT_CLASSES = {
    0: "Normal Pedestrian Motion",
    1: "Loitering / Lingering",
    2: "Carrying / Unloading Object",
    3: "Entering Vehicle",
    4: "Exiting Vehicle",
    5: "Unattended Package Drop"
}

def generate_live_inference_video(input_video_path, output_video_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # Initialize model and load trained weights
    model = CrowdContinualClassifier(num_classes=len(VIRAT_CLASSES), pretrained=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Using untrained model for layout testing.")
    model.eval()

    # Image transformations to match training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Setup OpenCV Video Capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video feed: {input_video_path}")
        return

    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup OpenCV Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_video_path}...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame every half second (assuming 30fps) to speed up CPU inference
        if frame_count % 15 == 0:
            # Convert OpenCV frame (BGR) to PIL Image (RGB) for PyTorch
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Prepare tensor
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                logits, _ = model(input_tensor)
                probs = nn.functional.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
            pred_idx = pred.item()
            confidence = conf.item()
            activity_label = VIRAT_CLASSES.get(pred_idx, "Unknown")
            risk_score = round(1.0 - confidence if pred_idx in [1, 5] else confidence * 0.4, 2)
            
            # Determine alert status
            alert_text = "ALERT: ANOMALY DETECTED" if risk_score > 0.55 else "STATUS: NORMAL"
            alert_color = (0, 0, 255) if risk_score > 0.55 else (0, 255, 0) # BGR format

        # --- Draw Overlays on the Frame ---
        
        # Draw background box for text readability
        cv2.rectangle(frame, (10, 10), (550, 150), (0, 0, 0), -1)
        
        # Overlay Text
        cv2.putText(frame, f"SYSTEM: Continual Learning Crowd Monitor", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"DETECTED: {activity_label} ({confidence*100:.1f}%)", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"RISK SCORE: {risk_score}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, alert_text, (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        # Write frame to output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Video saved successfully to: {output_video_path}")

if __name__ == "__main__":
    import os
    
    # Specify your exact chosen video
    target_video_name = "VIRAT_S_010005_04_000299_000323.mp4"
    input_video = os.path.join("raw_videos", target_video_name)
    
    # The output file will have "demo_" added to the front
    output_name = f"demo_{target_video_name}"
    model_weights = "continual_model.pth"
    
    if os.path.exists(input_video):
        print(f"Found target video: {input_video}")
        generate_live_inference_video(input_video, output_name, model_weights)
    else:
        print(f"Error: Could not find '{target_video_name}' in the 'raw_videos/' folder.")
        print("Please check the spelling and ensure the .mp4 file is in that folder.")