import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

from model import create_model
from dataset import get_transforms


class FrameClassifierInference:
    def __init__(self, model_path, device='cuda', img_size=224):
        """
        프레임 분류 추론기
        
        Args:
            model_path: 학습된 모델 경로
            device: 'cuda' or 'cpu'
            img_size: 입력 이미지 크기
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 모델 로드 (PyTorch 2.6+ 호환성)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        self.model = create_model(
            model_name=config.get('model_name', 'resnet50'),
            num_classes=config.get('num_classes', 3),
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = get_transforms(img_size=img_size, is_train=False)
        
        # 레이블 매핑
        self.label_names = ['OTE', 'Velum', 'None']
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
    
    def predict_frame(self, image):
        """
        단일 프레임 예측
        
        Args:
            image: PIL Image 또는 numpy array (BGR or RGB)
        
        Returns:
            predicted_class: 예측된 클래스 이름
            confidence: 신뢰도
            probabilities: 각 클래스별 확률
        """
        # numpy array를 PIL Image로 변환
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Transform 적용
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = self.label_names[predicted.item()]
        confidence = confidence.item()
        probs = probabilities.squeeze().cpu().numpy()
        
        return predicted_class, confidence, probs
    
    def predict_image(self, image_path):
        """이미지 파일 예측"""
        image = Image.open(image_path).convert('RGB')
        return self.predict_frame(image)
    
    def predict_video(self, video_path, output_path=None, 
                     frame_interval=1, visualize=True):
        """
        비디오 전체 프레임 예측
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 결과 저장 경로 (JSON)
            frame_interval: 프레임 추출 간격 (1=모든 프레임)
            visualize: 시각화 비디오 생성 여부
        
        Returns:
            results: 프레임별 예측 결과
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        results = {
            'video_path': str(video_path),
            'fps': fps,
            'total_frames': total_frames,
            'frame_predictions': []
        }
        
        # 시각화 비디오 작성기
        writer = None
        if visualize:
            output_video_path = Path(video_path).stem + '_predictions.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc='Processing video')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # 예측
                pred_class, confidence, probs = self.predict_frame(frame)
                
                # 결과 저장
                results['frame_predictions'].append({
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps,
                    'predicted_class': pred_class,
                    'confidence': float(confidence),
                    'probabilities': {
                        'OTE': float(probs[0]),
                        'Velum': float(probs[1]),
                        'None': float(probs[2])
                    }
                })
                
                # 시각화
                if visualize:
                    self._draw_prediction(frame, pred_class, confidence, probs)
                    writer.write(frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if writer:
            writer.release()
            print(f"Visualization video saved: {output_video_path}")
        
        # 통계 계산
        pred_counts = {'OTE': 0, 'Velum': 0, 'None': 0}
        for pred in results['frame_predictions']:
            pred_counts[pred['predicted_class']] += 1
        
        results['statistics'] = {
            'total_predicted_frames': len(results['frame_predictions']),
            'class_counts': pred_counts,
            'class_percentages': {
                k: v / len(results['frame_predictions']) * 100 
                for k, v in pred_counts.items()
            }
        }
        
        # 결과 저장
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        
        return results
    
    def _draw_prediction(self, frame, pred_class, confidence, probs):
        """프레임에 예측 결과 그리기"""
        # 배경 박스
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        
        # 예측 클래스
        text = f"Prediction: {pred_class}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        # 신뢰도
        text = f"Confidence: {confidence:.3f}"
        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # 클래스별 확률
        y_offset = 100
        for i, (label, prob) in enumerate(zip(self.label_names, probs)):
            text = f"{label}: {prob:.3f}"
            color = (0, 255, 0) if label == pred_class else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def analyze_video_segments(self, video_path, segment_duration=3.0):
        """
        비디오를 세그먼트로 나눠서 분석
        
        Args:
            video_path: 비디오 파일 경로
            segment_duration: 세그먼트 길이 (초)
        
        Returns:
            segment_results: 세그먼트별 분석 결과
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frames_per_segment = int(segment_duration * fps)
        num_segments = int(np.ceil(total_frames / frames_per_segment))
        
        segment_results = []
        
        for seg_idx in tqdm(range(num_segments), desc='Analyzing segments'):
            start_frame = seg_idx * frames_per_segment
            end_frame = min((seg_idx + 1) * frames_per_segment, total_frames)
            
            segment_predictions = []
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                pred_class, confidence, probs = self.predict_frame(frame)
                segment_predictions.append(pred_class)
            
            # 세그먼트 내 다수결 투표
            if segment_predictions:
                most_common = max(set(segment_predictions), 
                                 key=segment_predictions.count)
                confidence_rate = segment_predictions.count(most_common) / len(segment_predictions)
                
                segment_results.append({
                    'segment_index': seg_idx,
                    'start_time': start_frame / fps,
                    'end_time': end_frame / fps,
                    'predicted_class': most_common,
                    'confidence': confidence_rate,
                    'frame_count': len(segment_predictions)
                })
        
        cap.release()
        
        return segment_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Frame Classification Inference')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video or image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (JSON)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization video')
    parser.add_argument('--frame-interval', type=int, default=1,
                       help='Frame sampling interval')
    parser.add_argument('--segment-analysis', action='store_true',
                       help='Perform segment-based analysis')
    parser.add_argument('--segment-duration', type=float, default=3.0,
                       help='Duration of each segment in seconds')
    
    args = parser.parse_args()
    
    # 추론기 생성
    inferencer = FrameClassifierInference(
        model_path=args.model,
        device='cuda'
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # 비디오 추론
            if args.segment_analysis:
                results = inferencer.analyze_video_segments(
                    args.input,
                    segment_duration=args.segment_duration
                )
                print("\n=== Segment Analysis Results ===")
                for seg in results:
                    print(f"Segment {seg['segment_index']}: "
                          f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s | "
                          f"Predicted: {seg['predicted_class']} "
                          f"(Confidence: {seg['confidence']:.2%})")
            else:
                results = inferencer.predict_video(
                    args.input,
                    output_path=args.output,
                    frame_interval=args.frame_interval,
                    visualize=args.visualize
                )
                
                # 통계 출력
                print("\n=== Video Analysis Results ===")
                print(f"Total frames analyzed: {results['statistics']['total_predicted_frames']}")
                print("\nClass distribution:")
                for class_name, percentage in results['statistics']['class_percentages'].items():
                    count = results['statistics']['class_counts'][class_name]
                    print(f"  {class_name}: {count} frames ({percentage:.2f}%)")
        
        else:
            # 이미지 추론
            pred_class, confidence, probs = inferencer.predict_image(args.input)
            
            print("\n=== Image Classification Result ===")
            print(f"Predicted Class: {pred_class}")
            print(f"Confidence: {confidence:.4f}")
            print("\nClass Probabilities:")
            for label, prob in zip(inferencer.label_names, probs):
                print(f"  {label}: {prob:.4f}")


if __name__ == '__main__':
    main()
