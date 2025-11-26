"""
통합 수면 내시경 비디오 분석 파이프라인 (Fixed: Proper Crop + Color-based ROI)
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import timedelta
import json
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import sys

# Classification 모델 import
sys.path.append(str(Path(__file__).parent / 'ote_velum_classification_final'))
from model import create_model

class IntegratedDISEAnalyzer:
    def __init__(self, 
                 model_path,
                 fps_extract=1,
                 threshold_percent=10,
                 exclude_first_seconds=0,
                 exclude_last_seconds=0,
                 min_event_duration=1.0,
                 manual_max_area=None):
        
        self.fps_extract = fps_extract
        self.threshold_percent = threshold_percent
        self.exclude_first_seconds = exclude_first_seconds
        self.exclude_last_seconds = exclude_last_seconds
        self.min_event_duration = min_event_duration
        self.manual_max_area = manual_max_area
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEVICE: {self.device}")
        
        try:
            self.model = self._load_classification_model(model_path)
            print("✅ Classification 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['OTE', 'Velum', 'None'] 
        
        self.results = {
            'video_info': {},
            'frame_classifications': [],
            'segments': [],
            'occlusion_events': [],
            'summary': {},
            'max_area': manual_max_area if manual_max_area else 0,
            'max_area_frame': 0,
            'max_area_source': 'manual' if manual_max_area else 'auto'
        }
    
    def _load_classification_model(self, model_path):
        model = create_model('resnet50', num_classes=3, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def preprocess_frame(self, frame):
        """프레임 전처리 - 검은 배경 제거하고 내시경 영역만 CROP"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 안쪽으로 여유 두기 (vignetting 제거)
            margin_percent = 0.15  # 15% 안쪽으로
            margin_x = int(w * margin_percent)
            margin_y = int(h * margin_percent)
            
            x = max(0, x + margin_x)
            y = max(0, y + margin_y)
            w = min(frame.shape[1] - x, w - 2 * margin_x)
            h = min(frame.shape[0] - y, h - 2 * margin_y)
            
            # ✅ 핵심: 내시경 영역만 잘라내기 (검은 배경 없음)
            cropped = frame[y:y+h, x:x+w].copy()
            return cropped, (x, y, w, h)
        
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    
    def classify_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        return self.class_names[pred_idx], confidence
    
    def analyze_roi_dual_track(self, frame, label):
        """Color-based ROI 탐지 (이미 crop된 프레임에서 어두운 기도 영역 찾기)"""
        if label == 'None':
            return 0, None
        
        if frame is None or frame.size == 0:
            return 0, None
        
        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 어두운 영역 타겟팅
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 가장 큰 연결 영역
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels <= 1:
            return 0, None
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        area = float(stats[largest_label, cv2.CC_STAT_AREA])
        mask_final = (labels == largest_label).astype('uint8') * 255
        
        # 중심이 이미지 중앙 근처에 있는지 확인 (Velum은 더 관대하게)
        h, w = frame.shape[:2]
        center_x, center_y = centroids[largest_label]
        img_center_x, img_center_y = w / 2, h / 2
        distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        
        # Velum은 중심 검증을 더 관대하게 (50% vs 45%)
        max_distance_ratio = 0.50 if label == 'Velum' else 0.45
        max_distance = min(w, h) * max_distance_ratio
        
        if distance_from_center > max_distance:
            return 0, None
        
        # OTE 레이블 특화 필터링만 적용
        if label == 'OTE':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 80)
            edge_score = np.mean(edges)
            if edge_score > 12.0 and area < 3000:
                return 0, None
            roi_pixels = gray[mask_final > 0]
            if len(roi_pixels) > 0 and np.std(roi_pixels) > 50 and area < 1000:
                return 0, None

        return area, mask_final

    def detect_segments(self, frame_classifications):
        if not frame_classifications: return []
        smoothed_labels = [f['label'] for f in frame_classifications]
        for i in range(1, len(smoothed_labels) - 1):
            if smoothed_labels[i-1] == smoothed_labels[i+1] and smoothed_labels[i] != smoothed_labels[i-1]:
                smoothed_labels[i] = smoothed_labels[i-1]
                frame_classifications[i]['label'] = smoothed_labels[i-1]
        
        segments = []
        current_segment = None
        for i, frame_data in enumerate(frame_classifications):
            label = frame_data['label']
            if label in ['OTE', 'Velum']:
                if current_segment is None:
                    current_segment = {'label': label, 'start_frame': frame_data['frame_number'], 'start_time': frame_data['timestamp'], 'frames': [frame_data]}
                elif current_segment['label'] == label:
                    current_segment['frames'].append(frame_data)
                else:
                    current_segment['end_frame'] = current_segment['frames'][-1]['frame_number']
                    current_segment['end_time'] = current_segment['frames'][-1]['timestamp']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                    segments.append(current_segment)
                    current_segment = {'label': label, 'start_frame': frame_data['frame_number'], 'start_time': frame_data['timestamp'], 'frames': [frame_data]}
            else:
                if current_segment is not None:
                    current_segment['end_frame'] = current_segment['frames'][-1]['frame_number']
                    current_segment['end_time'] = current_segment['frames'][-1]['timestamp']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                    segments.append(current_segment)
                    current_segment = None
        
        if current_segment is not None:
            current_segment['end_frame'] = current_segment['frames'][-1]['frame_number']
            current_segment['end_time'] = current_segment['frames'][-1]['timestamp']
            current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
            segments.append(current_segment)
        
        # ✅ 레이블별로 전체 프레임에서 max_area 계산
        valid_segments = [s for s in segments if s['duration'] > 0.5]
        
        label_max_areas = {}  # {'OTE': max_area, 'Velum': max_area}
        
        for label in ['OTE', 'Velum']:
            # 해당 레이블의 모든 프레임 수집
            label_frames = [f for f in frame_classifications if f['label'] == label]
            label_areas = [f['roi_area'] for f in label_frames if f.get('roi_area', 0) > 0]
            
            if label_areas:
                label_max_areas[label] = max(label_areas)
                # 해당 max_area를 가진 프레임 찾기
                max_frame = max(label_frames, key=lambda f: f.get('roi_area', 0))
                print(f"  📍 {label} global max_area = {label_max_areas[label]:.0f} px² (frame {max_frame['frame_number']})")
            else:
                label_max_areas[label] = 0
        
        # 각 segment에 해당 레이블의 max_area 할당
        for segment in valid_segments:
            segment['max_area'] = label_max_areas.get(segment['label'], 0)
            # max_area_frame은 전체 레이블의 max_area 프레임
            if segment['max_area'] > 0:
                label_frames = [f for f in frame_classifications if f['label'] == segment['label']]
                max_frame = max(label_frames, key=lambda f: f.get('roi_area', 0))
                segment['max_area_frame'] = max_frame['frame_number']
            else:
                segment['max_area_frame'] = segment['start_frame']
            
        return valid_segments
    
    def detect_occlusion_events(self, segments):
        """Segment별 max_area 기준으로 폐쇄 이벤트 감지"""
        events = []
        
        for segment in segments:
            segment_max_area = segment.get('max_area', 0)
            
            if segment_max_area < 1000:
                print(f"⚠️ {segment['label']} segment max_area ({segment_max_area:.0f}) too small, skipping")
                continue
            
            # 이 segment의 threshold
            threshold_area = segment_max_area * (1 - self.threshold_percent / 100)
            print(f"  🎯 {segment['label']} threshold: {threshold_area:.0f} px² ({self.threshold_percent}% of {segment_max_area:.0f})")
            
            current_event = None
            
            for frame_data in segment['frames']:
                roi_area = frame_data.get('roi_area', 0)
                
                # Segment 기준으로 감소율 계산
                if roi_area > 0:
                    frame_data['reduction_percent'] = (1 - roi_area / segment_max_area) * 100
                else:
                    frame_data['reduction_percent'] = 100.0
                
                # ✅ 수정: roi_area가 threshold보다 작거나 0인 경우 모두 이벤트로 간주
                is_occlusion = (roi_area < threshold_area)  # 0도 포함
                
                if is_occlusion:
                    area_reduction = (1 - roi_area / segment_max_area) * 100 if roi_area > 0 else 100.0
                    severity = self._classify_severity(area_reduction)
                    if current_event is None:
                        current_event = {
                            'segment_label': segment['label'], 
                            'segment_max_area': segment_max_area,
                            'start_frame': frame_data['frame_number'], 
                            'start_time': frame_data['timestamp'], 
                            'severity': severity, 
                            'max_reduction': area_reduction,
                            'frames': [frame_data]
                        }
                    else:
                        current_event['frames'].append(frame_data)
                        current_event['max_reduction'] = max(current_event['max_reduction'], area_reduction)
                        if self._severity_level(severity) > self._severity_level(current_event['severity']):
                            current_event['severity'] = severity
                else:
                    if current_event is not None:
                        current_event['end_frame'] = current_event['frames'][-1]['frame_number']
                        current_event['end_time'] = current_event['frames'][-1]['timestamp']
                        current_event['duration'] = current_event['end_time'] - current_event['start_time']
                        if current_event['duration'] >= self.min_event_duration: 
                            events.append(current_event)
                            print(f"    ✓ Event detected: {current_event['severity']} at {current_event['start_time']:.1f}s-{current_event['end_time']:.1f}s ({current_event['max_reduction']:.1f}% reduction)")
                        else:
                            print(f"    ✗ Event too short: {current_event['duration']:.2f}s < {self.min_event_duration}s")
                        current_event = None
            
            if current_event is not None:
                current_event['end_frame'] = current_event['frames'][-1]['frame_number']
                current_event['end_time'] = current_event['frames'][-1]['timestamp']
                current_event['duration'] = current_event['end_time'] - current_event['start_time']
                if current_event['duration'] >= self.min_event_duration: 
                    events.append(current_event)
                    print(f"    ✓ Event detected (at end): {current_event['severity']} at {current_event['start_time']:.1f}s-{current_event['end_time']:.1f}s ({current_event['max_reduction']:.1f}% reduction)")
                else:
                    print(f"    ✗ Event too short (at end): {current_event['duration']:.2f}s < {self.min_event_duration}s")
        
        return events
    
    def _classify_severity(self, reduction_percent):
        if reduction_percent >= 70: return 'Critical'
        if reduction_percent >= 50: return 'Severe'
        if reduction_percent >= 30: return 'Moderate'
        return 'Mild'
    
    def _severity_level(self, severity):
        return {'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4}.get(severity, 0)
    
    def _create_segment_reference_images(self, video_path, output_dir):
        """각 segment별 reference 이미지 생성 (OTE, Velum 각각)"""
        cap = cv2.VideoCapture(str(video_path))
        
        for segment in self.results['segments']:
            if segment.get('max_area', 0) == 0:
                continue
                
            max_area_frame = segment.get('max_area_frame', 0)
            if max_area_frame == 0:
                continue
            
            # 비디오에서 해당 프레임 읽기
            cap.set(cv2.CAP_PROP_POS_FRAMES, max_area_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 전처리 (내시경 영역만 crop)
            preprocessed, bbox = self.preprocess_frame(frame)
            
            # ROI 탐지
            label = segment['label']
            _, roi_mask = self.analyze_roi_dual_track(preprocessed, label)
            
            overlay = preprocessed.copy()
            
            # ROI 윤곽선만 표시
            if roi_mask is not None:
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 255), 4)
            
            # segment 정보 텍스트 추가
            cv2.rectangle(overlay, (5, 5), (overlay.shape[1]-5, 100), (0, 0, 0), -1)
            cv2.rectangle(overlay, (5, 5), (overlay.shape[1]-5, 100), (255, 255, 255), 2)
            
            segment_color = (255, 255, 0) if label == 'OTE' else (255, 0, 255)
            cv2.putText(overlay, f"Reference: {label}", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, segment_color, 2)
            cv2.putText(overlay, f"Frame: {max_area_frame}", (15, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Max Area: {segment['max_area']:.0f} px²", (15, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s", (15, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            ref_path = Path(output_dir) / "overlays" / f"reference_{label}.jpg"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ref_path), overlay)
            
            # results에 저장
            if 'reference_images' not in self.results:
                self.results['reference_images'] = {}
            self.results['reference_images'][label] = str(ref_path)
            
            print(f"✅ {label} Reference 이미지 생성: {ref_path}")
        
        cap.release()
    
    def _create_debug_frames(self, video_path, output_dir):
        """각 프레임별로 ROI 윤곽선과 SEGMENT 정보를 표시한 디버깅 이미지 생성"""
        debug_dir = Path(output_dir) / "debug_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        reference_max_area = self.results['max_area']
        
        saved_count = 0
        for frame_data in tqdm(self.results['frame_classifications'], desc="Debug frames"):
            frame_num = frame_data['frame_number']
            label = frame_data['label']
            roi_area = frame_data.get('roi_area', 0)
            timestamp = frame_data['timestamp']
            
            # 해당 프레임 읽기
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 전처리 (내시경 영역만 crop)
            preprocessed, bbox = self.preprocess_frame(frame)
            
            # ROI 탐지
            roi_mask = None
            if label in ['OTE', 'Velum']:
                _, roi_mask = self.analyze_roi_dual_track(preprocessed, label)
            
            # 디버깅 이미지는 preprocessed 기준
            debug_frame = preprocessed.copy()
            
            # ROI 윤곽선만 표시
            if roi_mask is not None and roi_area > 0:
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_frame, contours, -1, (0, 255, 255), 3)
            
            # 해당 프레임이 속한 segment 찾기
            current_segment = None
            for segment in self.results['segments']:
                if segment['start_frame'] <= frame_num <= segment['end_frame']:
                    current_segment = segment
                    break
            
            # Reduction 계산
            reduction = (1 - roi_area / reference_max_area) * 100 if (reference_max_area > 0 and roi_area > 0) else 0
            
            # 정보 패널 (상단)
            info_height = 140
            cv2.rectangle(debug_frame, (5, 5), (debug_frame.shape[1]-5, info_height), (0, 0, 0), -1)
            cv2.rectangle(debug_frame, (5, 5), (debug_frame.shape[1]-5, info_height), (255, 255, 255), 2)
            
            # 프레임 정보
            font_scale = 0.7
            thickness = 2
            y_offset = 25
            line_height = 25
            
            cv2.putText(debug_frame, f"Frame: {frame_num} | Time: {timestamp:.2f}s", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            y_offset += line_height
            
            # Label 정보
            label_color = (0, 255, 255) if label == 'OTE' else (255, 0, 255) if label == 'Velum' else (128, 128, 128)
            cv2.putText(debug_frame, f"Label: {label}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
            y_offset += line_height
            
            # ROI 정보
            if roi_area > 0:
                cv2.putText(debug_frame, f"ROI Area: {roi_area:.0f} px²", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
                y_offset += line_height
                cv2.putText(debug_frame, f"Reduction: {reduction:.1f}%", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            else:
                cv2.putText(debug_frame, f"ROI Area: N/A", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 128, 128), thickness)
            
            # Segment 정보 (하단)
            if current_segment:
                segment_y = debug_frame.shape[0] - 80
                cv2.rectangle(debug_frame, (5, segment_y), (debug_frame.shape[1]-5, debug_frame.shape[0]-5), (0, 0, 0), -1)
                cv2.rectangle(debug_frame, (5, segment_y), (debug_frame.shape[1]-5, debug_frame.shape[0]-5), (255, 255, 0), 3)
                
                segment_color = (255, 255, 0) if current_segment['label'] == 'OTE' else (255, 0, 255)
                cv2.putText(debug_frame, f"SEGMENT: {current_segment['label']}", (15, segment_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, segment_color, 2)
                cv2.putText(debug_frame, f"Time: {current_segment['start_time']:.1f}s - {current_segment['end_time']:.1f}s", (15, segment_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                segment_y = debug_frame.shape[0] - 50
                cv2.rectangle(debug_frame, (5, segment_y), (debug_frame.shape[1]-5, debug_frame.shape[0]-5), (0, 0, 0), -1)
                cv2.putText(debug_frame, "SEGMENT: None", (15, segment_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            
            # 파일 저장
            filename = f"frame_{frame_num:06d}_t{timestamp:.2f}s_{label}.jpg"
            filepath = debug_dir / filename
            cv2.imwrite(str(filepath), debug_frame)
            saved_count += 1
        
        cap.release()
        print(f"✅ 디버깅 이미지 {saved_count}개 저장 완료: {debug_dir}")
    
    def create_event_clips(self, video_path, events, output_dir):
        """Side-by-Side 비디오: 우측에 ROI 윤곽선 + segment 정보 표시"""
        output_dir = Path(output_dir)
        clips_dir = output_dir / 'event_clips'
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # H.264 코덱 사용 (브라우저 호환성 개선)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        ext = 'mp4'
        print(f"🎥 Using H.264 codec for better browser compatibility")

        segments = self.results['segments']

        for i, event in enumerate(events):
            start_frame = max(0, event['start_frame'] - int(fps * 1.0))
            end_frame = event['end_frame'] + int(fps * 1.0)
            
            filename = f"event_{i+1:02d}_{event['segment_label']}_{event['severity']}.{ext}"
            filepath = clips_dir / filename
            
            # 첫 프레임으로 크기 결정
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, first_frame = cap.read()
            if not ret:
                continue
                
            preprocessed_first, _ = self.preprocess_frame(first_frame)
            h_crop, w_crop = preprocessed_first.shape[:2]
            
            # Side-by-side 크기
            out_w = w_crop * 2
            out_h = h_crop
            
            out = cv2.VideoWriter(str(filepath), fourcc, fps, (out_w, out_h))
            if not out.isOpened():
                print(f"⚠️ H.264 코덱 실패, mp4v로 재시도: {filepath}")
                fourcc_fallback = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(filepath), fourcc_fallback, fps, (out_w, out_h))
                if not out.isOpened():
                    print(f"❌ 비디오 Writer 초기화 실패: {filepath}")
                    continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = 0
            
            # 해당 이벤트의 segment 정보
            event_segment = None
            for segment in segments:
                if segment['label'] == event['segment_label'] and \
                   segment['start_frame'] <= event['start_frame'] <= segment['end_frame']:
                    event_segment = segment
                    break
            
            for f_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret: break
                
                # 전처리
                preprocessed, bbox = self.preprocess_frame(frame)
                
                left = preprocessed.copy()
                right = preprocessed.copy()
                
                # 현재 프레임이 속한 segment 찾기
                current_segment = None
                for segment in segments:
                    if segment['start_frame'] <= f_idx <= segment['end_frame']:
                        current_segment = segment
                        break
                
                # ✅ 실시간 ROI 계산 및 reduction 계산
                reduction = 0
                if current_segment and current_segment['label'] in ['OTE', 'Velum']:
                    roi_area_current, roi_mask = self.analyze_roi_dual_track(preprocessed, current_segment['label'])
                    
                    # ROI 윤곽선 표시
                    if roi_mask is not None:
                        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(right, contours, -1, (0, 255, 255), 3)
                    
                    # Reduction 계산 (segment의 max_area 기준)
                    segment_max_area = current_segment.get('max_area', 0)
                    if segment_max_area > 0:
                        if roi_area_current > 0:
                            reduction = (1 - roi_area_current / segment_max_area) * 100
                        else:
                            reduction = 100.0
                
                # 이벤트 구간 여부
                is_event = event['start_frame'] <= f_idx <= event['end_frame']
                
                # 상단 정보
                info_height = 95
                cv2.rectangle(right, (5, 5), (w_crop-5, info_height), (0, 0, 0), -1)
                cv2.rectangle(right, (5, 5), (w_crop-5, info_height), (255, 255, 255), 2)
                
                text_color = (0, 0, 255) if is_event else (255, 255, 255)
                font_scale = 1.0
                thickness = 2
                
                if current_segment:
                    cv2.putText(right, f"Segment: {current_segment['label']}", (15, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                    cv2.putText(right, f"Max: {current_segment.get('max_area', 0):.0f} px²", (15, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                    cv2.putText(right, f"Reduction: {reduction:.1f}%", (15, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                else:
                    cv2.putText(right, "Segment: None", (15, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                
                # 이벤트 구간 표시 (하단)
                if is_event:
                    event_height = 85
                    cv2.rectangle(right, (5, h_crop-event_height), (w_crop-5, h_crop-5), (0, 0, 0), -1)
                    cv2.rectangle(right, (5, h_crop-event_height), (w_crop-5, h_crop-5), (0, 0, 255), 4)
                    cv2.putText(right, f"OCCLUSION EVENT: {event['severity']}", (15, h_crop-45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(right, f"Reduction: {reduction:.1f}%", (15, h_crop-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                combined = np.hstack((left, right))
                out.write(combined)
                frame_count += 1
            
            out.release()
            if filepath.exists() and filepath.stat().st_size > 0:
                print(f"✅ 클립 생성 완료: {filename} ({frame_count} 프레임)")
                event['clip_path'] = f"event_clips/{filename}"
            else:
                print(f"❌ 클립 생성 실패: {filename}")
                
        cap.release()

    def analyze_video(self, video_path, output_dir=None):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        start_frame = int(fps * self.exclude_first_seconds)
        end_frame = total_frames - int(fps * self.exclude_last_seconds)
        
        self.results['video_info'] = {'filename': video_path.name, 'fps': fps, 'total_frames': total_frames, 'duration': duration}
        print(f"\n🚀 분석 시작: {video_path.name}")
        
        if self.manual_max_area:
            print(f"📌 Manual max_area 사용: {self.manual_max_area:.0f} px²")
        
        interval = max(1, int(fps / self.fps_extract))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        
        pbar = tqdm(total=(end_frame - start_frame), desc="Processing")
        
        while curr_frame < end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            if (curr_frame - start_frame) % interval == 0:
                preprocessed, bbox = self.preprocess_frame(frame)
                label, confidence = self.classify_frame(preprocessed)
                roi_area = 0
                if label in ['OTE', 'Velum']:
                    roi_area, _ = self.analyze_roi_dual_track(preprocessed, label)
                
                if not self.manual_max_area:
                    if roi_area > self.results['max_area']:
                        self.results['max_area'] = roi_area
                        self.results['max_area_frame'] = curr_frame

                timestamp = curr_frame / fps
                self.results['frame_classifications'].append({
                    'frame_number': curr_frame, 'timestamp': timestamp,
                    'label': label, 'roi_area': roi_area
                })
                
            curr_frame += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        
        print(f"📊 Reference max_area: {self.results['max_area']:.0f} px² ({self.results['max_area_source']})")
        
        print("🔍 후처리 중 (구간 병합 및 이벤트 감지)...")
        self.results['segments'] = self.detect_segments(self.results['frame_classifications'])
        
        # ✅ segment_references 저장 (리포트용)
        self.results['segment_references'] = {}
        for segment in self.results['segments']:
            if segment.get('max_area', 0) > 0:
                self.results['segment_references'][segment['label']] = {
                    'max_area': segment['max_area'],
                    'frame_number': segment['max_area_frame']
                }
        
        self.results['occlusion_events'] = self.detect_occlusion_events(self.results['segments'])
        
        self.results['metadata'] = {'threshold_percent': self.threshold_percent}
        
        self.results['summary'] = {
            'total_segments': len(self.results['segments']),
            'ote_segments': sum(1 for s in self.results['segments'] if s['label'] == 'OTE'),
            'velum_segments': sum(1 for s in self.results['segments'] if s['label'] == 'Velum'),
            'total_events': len(self.results['occlusion_events']),
            'events_by_severity': {
                'Critical': sum(1 for e in self.results['occlusion_events'] if e['severity'] == 'Critical'),
                'Severe': sum(1 for e in self.results['occlusion_events'] if e['severity'] == 'Severe'),
                'Moderate': sum(1 for e in self.results['occlusion_events'] if e['severity'] == 'Moderate'),
                'Mild': sum(1 for e in self.results['occlusion_events'] if e['severity'] == 'Mild'),
            }
        }
        
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Segment별 reference 이미지 생성
            if self.results['segments']:
                print("📸 Segment별 Reference 이미지 생성 중...")
                self._create_segment_reference_images(video_path, output_dir)
            
            print("🔍 디버깅용 프레임별 이미지 생성 중...")
            self._create_debug_frames(video_path, output_dir)
            
            if self.results['occlusion_events']:
                print("🎥 이벤트 클립 생성 중...")
                self.create_event_clips(video_path, self.results['occlusion_events'], output_dir)
            
            with open(out_path / 'analysis_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
                
        print("✅ 분석 완료!")
        return self.results