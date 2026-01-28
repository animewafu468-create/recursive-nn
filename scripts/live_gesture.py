#!/usr/bin/env python3
"""
Live Hand Gesture Recognition using webcam.

Usage:
    python scripts/live_gesture.py --checkpoint ./checkpoints/hagrid/gen_002/best.pt

Controls:
    q - Quit
    s - Save screenshot
    r - Reset detection smoothing
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from models import ResNet18
from data.loaders import HAGRID_CLASSES, HAGRID_EMOJIS


class GesturePredictor:
    """Real-time gesture prediction with smoothing."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        smoothing_window: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = ResNet18(num_classes=len(HAGRID_CLASSES))

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

        # Smoothing buffer for stable predictions
        self.prediction_buffer = deque(maxlen=smoothing_window)

        # Preprocessing
        self.image_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to model input tensor."""
        # Resize
        frame = cv2.resize(frame, (self.image_size, self.image_size))

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To tensor [0, 1]
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        # Normalize
        tensor = (tensor - self.mean) / self.std

        return tensor

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Predict gesture from frame.

        Returns:
            gesture_name: Predicted gesture (or "no_gesture" if below threshold)
            confidence: Confidence score
            probabilities: All class probabilities
        """
        # Preprocess
        tensor = self.preprocess(frame)

        # Forward pass
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Add to smoothing buffer
        self.prediction_buffer.append(probs)

        # Average over buffer
        smoothed_probs = np.mean(self.prediction_buffer, axis=0)

        # Get prediction
        pred_idx = np.argmax(smoothed_probs)
        confidence = smoothed_probs[pred_idx]

        if confidence < self.confidence_threshold:
            return "no_gesture", confidence, smoothed_probs

        gesture_name = HAGRID_CLASSES[pred_idx]
        return gesture_name, confidence, smoothed_probs

    def reset(self):
        """Reset smoothing buffer."""
        self.prediction_buffer.clear()


def draw_prediction(
    frame: np.ndarray,
    gesture: str,
    confidence: float,
    fps: float,
    probs: np.ndarray = None,
) -> np.ndarray:
    """Draw prediction overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Gesture text
    emoji = HAGRID_EMOJIS.get(gesture, "❓")
    text = f"{gesture.upper()}" if gesture != "no_gesture" else "NO GESTURE"
    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)

    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Confidence bar
    bar_width = int(300 * confidence)
    cv2.rectangle(frame, (20, 130), (20 + bar_width, 145), color, -1)
    cv2.rectangle(frame, (20, 130), (320, 145), (100, 100, 100), 2)

    # Top 3 predictions (right side)
    if probs is not None:
        top_indices = np.argsort(probs)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            y_pos = 180 + i * 30
            name = HAGRID_CLASSES[idx]
            prob = probs[idx]
            cv2.putText(frame, f"{name}: {prob:.1%}", (w - 200, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Live hand gesture recognition")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=5,
        help="Number of frames to smooth predictions over",
    )
    args = parser.parse_args()

    # Initialize predictor
    predictor = GesturePredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        smoothing_window=args.smoothing,
        confidence_threshold=args.threshold,
    )

    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("🖐️  LIVE GESTURE RECOGNITION")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  r - Reset smoothing")
    print("=" * 50 + "\n")

    frame_count = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip horizontally (mirror mode)
            frame = cv2.flip(frame, 1)

            # Predict
            gesture, confidence, probs = predictor.predict(frame)

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed

            # Draw overlay
            display_frame = draw_prediction(frame, gesture, confidence, fps, probs)

            # Show frame
            cv2.imshow("Hand Gesture Recognition", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"gesture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('r'):
                predictor.reset()
                print("Reset prediction buffer")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    main()
