"""
통합 수면 내시경 비디오 분석 파이프라인
1. OTE/Velum/None 분류
2. OTE/Velum 구간에서만 폐색 분석
3. 연속 구간 감지
4. 위험 구간 비디오 클립 생성
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

# Classification 모델 import (경로 조정 필요)
sys.path.append(str(Path(__file__).parent / 'ote_velum_classification_final'))
from model import get_model


class IntegratedDISEAnalyzer:
    """통합 DISE 비디오 분석기"""
    
    def __init__(self, 
                 model_path,
                 fps_extract=5,
                 threshold_percent=30,
                 exclude_first_seconds=2,
                 exclude_last_seconds=3,
                 min_event_duration=1.0):
        """
        Args:
            model_path: Classification 모델 경로
            fps_extract: 초당 추출할 프레임 수
            threshold_percent: 폐색 기준 (%)
            exclude_first_seconds: 앞부분 제외 (초)
            exclude_last_seconds: 뒷부분 제외 (초)
            min_event_duration: 최소 이벤트 지속 시간 (초)
        """
        self.fps_extract = fps_extract
        self.threshold_percent = threshold_percent
        self.exclude_first_seconds = exclude_first_seconds
        self.exclude_last_seconds = exclude_last_seconds
        self.min_event_duration = min_event_duration
        
        # Classification 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_classification_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['None', 'OTE', 'Velum']
        
        self.results = {
            'video_info': {},
            'frame_classifications': [],
            'segments': [],  # OTE/Velum 연속 구간
            'occlusion_events': [],  # 폐색 이벤트 (연속 구간)
            'summary': {}
        }
    
    def _load_classification_model(self, model_path):
        """Classification 모델 로드"""
        model = get_model('resnet50', num_classes=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def preprocess_frame(self, frame):
        """프레임 전처리 - 검정 배경만 제거 (크롭 없이)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 여유 두기 (모서리 제거)
            margin_percent = 0.075
            margin_x = int(w * margin_percent)
            margin_y = int(h * margin_percent)
            
            x = max(0, x + margin_x)
            y = max(0, y + margin_y)
            w = min(frame.shape[1] - x, w - 2 * margin_x)
            h = min(frame.shape[0] - y, h - 2 * margin_y)
            
            # 전체 프레임 크기 유지, 검정 배경만 마스킹
            result = np.zeros_like(frame)
            result[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
            
            return result, (x, y, w, h)
        
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    
    def classify_frame(self, frame):
        """프레임 분류 (OTE/Velum/None)"""
        # PIL Image로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform & 추론
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        return self.class_names[pred_class], confidence
    
    def extract_roi_area(self, frame):
        """ROI 영역 면적 추출 (기도 내부 어두운 영역)"""
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
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            area = stats[largest_label, cv2.CC_STAT_AREA]
            mask_final = (labels == largest_label).astype('uint8') * 255
            return float(area), mask_final
        
        return 0, None
    
    def detect_segments(self, frame_classifications):
        """연속 구간 감지 (OTE/Velum)"""
        segments = []
        current_segment = None
        
        for i, frame_data in enumerate(frame_classifications):
            label = frame_data['label']
            
            if label in ['OTE', 'Velum']:
                if current_segment is None:
                    # 새 구간 시작
                    current_segment = {
                        'label': label,
                        'start_frame': frame_data['frame_number'],
                        'start_time': frame_data['timestamp'],
                        'frames': [frame_data]
                    }
                elif current_segment['label'] == label:
                    # 같은 라벨 계속
                    current_segment['frames'].append(frame_data)
                else:
                    # 라벨 변경 -> 이전 구간 종료
                    current_segment['end_frame'] = current_segment['frames'][-1]['frame_number']
                    current_segment['end_time'] = current_segment['frames'][-1]['timestamp']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                    segments.append(current_segment)
                    
                    # 새 구간 시작
                    current_segment = {
                        'label': label,
                        'start_frame': frame_data['frame_number'],
                        'start_time': frame_data['timestamp'],
                        'frames': [frame_data]
                    }
            else:
                # None -> 구간 종료
                if current_segment is not None:
                    current_segment['end_frame'] = current_segment['frames'][-1]['frame_number']
                    current_segment['end_time'] = current_segment['frames'][-1]['timestamp']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                    segments.append(current_segment)
                    current_segment = None
        
        # 마지막 구간 처리
        if current_segment is not None:
            current_segment['end_frame'] = current_segment['frames'][-1]['frame_number']
            current_segment['end_time'] = current_segment['frames'][-1]['timestamp']
            current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
            segments.append(current_segment)
        
        return segments
    
    def detect_occlusion_events(self, segments):
        """폐색 이벤트 감지 (연속 구간)"""
        events = []
        
        for segment in segments:
            # 해당 구간의 최대 ROI 면적
            max_area = max([f.get('roi_area', 0) for f in segment['frames']])
            
            if max_area == 0:
                continue
            
            threshold_area = max_area * (1 - self.threshold_percent / 100)
            
            # 폐색 구간 감지
            current_event = None
            
            for frame_data in segment['frames']:
                roi_area = frame_data.get('roi_area', 0)
                
                if roi_area > 0:
                    area_reduction = (1 - roi_area / max_area) * 100
                    
                    if roi_area < threshold_area:
                        # 폐색 상태
                        severity = self._classify_severity(area_reduction)
                        
                        if current_event is None:
                            # 새 이벤트 시작
                            current_event = {
                                'segment_label': segment['label'],
                                'start_frame': frame_data['frame_number'],
                                'start_time': frame_data['timestamp'],
                                'severity': severity,
                                'max_reduction': area_reduction,
                                'frames': [frame_data]
                            }
                        else:
                            # 이벤트 계속
                            current_event['frames'].append(frame_data)
                            current_event['max_reduction'] = max(
                                current_event['max_reduction'], 
                                area_reduction
                            )
                            # 심각도 업데이트 (더 심각한 것으로)
                            if self._severity_level(severity) > self._severity_level(current_event['severity']):
                                current_event['severity'] = severity
                    else:
                        # 폐색 해제
                        if current_event is not None:
                            current_event['end_frame'] = current_event['frames'][-1]['frame_number']
                            current_event['end_time'] = current_event['frames'][-1]['timestamp']
                            current_event['duration'] = current_event['end_time'] - current_event['start_time']
                            
                            # 최소 지속 시간 체크
                            if current_event['duration'] >= self.min_event_duration:
                                events.append(current_event)
                            
                            current_event = None
            
            # 마지막 이벤트 처리
            if current_event is not None:
                current_event['end_frame'] = current_event['frames'][-1]['frame_number']
                current_event['end_time'] = current_event['frames'][-1]['timestamp']
                current_event['duration'] = current_event['end_time'] - current_event['start_time']
                
                if current_event['duration'] >= self.min_event_duration:
                    events.append(current_event)
        
        return events
    
    def _classify_severity(self, reduction_percent):
        """폐색 심각도 분류"""
        if reduction_percent >= 70:
            return 'Critical'
        elif reduction_percent >= 50:
            return 'Severe'
        elif reduction_percent >= 30:
            return 'Moderate'
        else:
            return 'Mild'
    
    def _severity_level(self, severity):
        """심각도 숫자 변환"""
        levels = {'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4}
        return levels.get(severity, 0)
    
    def create_event_clips(self, video_path, events, output_dir):
        """위험 구간 비디오 클립 생성"""
        output_dir = Path(output_dir)
        clips_dir = output_dir / 'event_clips'
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        clip_paths = []
        
        for i, event in enumerate(events):
            # 클립 범위 설정 (전후 0.5초 여유)
            start_frame = max(0, event['start_frame'] - int(fps * 0.5))
            end_frame = event['end_frame'] + int(fps * 0.5)
            
            clip_filename = f"event_{i+1:03d}_{event['segment_label']}_{event['severity']}.mp4"
            clip_path = clips_dir / clip_filename
            
            # VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            
            # 프레임 추출 및 저장
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 위험 구간 표시
                if event['start_frame'] <= frame_num <= event['end_frame']:
                    # 빨간 테두리
                    cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 5)
                    
                    # 텍스트 표시
                    text = f"{event['severity']} - {event['segment_label']}"
                    cv2.putText(frame, text, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                out.write(frame)
            
            out.release()
            
            event['clip_path'] = str(clip_path)
            clip_paths.append(clip_path)
        
        cap.release()
        
        return clip_paths
    
    def analyze_video(self, video_path, output_dir=None):
        """
        비디오 통합 분석
        
        Returns:
            results: {
                'video_info': {...},
                'frame_classifications': [...],
                'segments': [...],
                'occlusion_events': [...],
                'summary': {...}
            }
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"비디오를 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        
        # 처리 범위
        exclude_first_frames = int(fps * self.exclude_first_seconds)
        exclude_last_frames = int(fps * self.exclude_last_seconds)
        start_frame = exclude_first_frames
        end_frame = total_frames - exclude_last_frames
        
        self.results['video_info'] = {
            'filename': video_path.name,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'analyzed_range': [start_frame, end_frame]
        }
        
        print("\n" + "="*70)
        print(f"📹 비디오 분석: {video_path.name}")
        print("="*70)
        print(f"  - FPS: {fps:.2f}")
        print(f"  - 총 프레임: {total_frames} ({duration:.1f}초)")
        print(f"  - 분석 구간: {start_frame} ~ {end_frame}")
        
        # 프레임 추출 간격
        frame_interval = max(1, int(fps / self.fps_extract))
        
        # 1단계: 프레임별 분류 + ROI 분석
        print("\n🔍 1단계: 프레임 분류 및 ROI 분석...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        
        pbar = tqdm(total=(end_frame - start_frame), desc="프레임 처리")
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_count - start_frame) % frame_interval == 0:
                # 전처리
                preprocessed, bbox = self.preprocess_frame(frame)
                
                # 분류
                label, confidence = self.classify_frame(preprocessed)
                
                # ROI 면적 (OTE/Velum인 경우만)
                roi_area = 0
                if label in ['OTE', 'Velum']:
                    roi_area, _ = self.extract_roi_area(preprocessed)
                
                timestamp = frame_count / fps
                
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'time_str': str(timedelta(seconds=int(timestamp))),
                    'label': label,
                    'confidence': confidence,
                    'roi_area': roi_area,
                    'bbox': bbox
                }
                
                self.results['frame_classifications'].append(frame_data)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # 2단계: 연속 구간 감지
        print("\n🔍 2단계: 연속 구간 감지...")
        self.results['segments'] = self.detect_segments(
            self.results['frame_classifications']
        )
        
        print(f"  ✓ 총 {len(self.results['segments'])}개 구간 감지")
        for seg in self.results['segments']:
            print(f"    - {seg['label']}: {seg['start_time']:.1f}s ~ {seg['end_time']:.1f}s "
                  f"({seg['duration']:.1f}s)")
        
        # 3단계: 폐색 이벤트 감지
        print("\n🔍 3단계: 폐색 이벤트 감지...")
        self.results['occlusion_events'] = self.detect_occlusion_events(
            self.results['segments']
        )
        
        print(f"  ✓ 총 {len(self.results['occlusion_events'])}개 이벤트 감지")
        for i, event in enumerate(self.results['occlusion_events'], 1):
            print(f"    #{i} {event['segment_label']} - {event['severity']}: "
                  f"{event['start_time']:.1f}s ~ {event['end_time']:.1f}s "
                  f"({event['duration']:.1f}s, {event['max_reduction']:.1f}% 감소)")
        
        # 4단계: 비디오 클립 생성
        if output_dir and self.results['occlusion_events']:
            print("\n🔍 4단계: 위험 구간 비디오 클립 생성...")
            clip_paths = self.create_event_clips(
                video_path, 
                self.results['occlusion_events'], 
                output_dir
            )
            print(f"  ✓ {len(clip_paths)}개 클립 생성 완료")
        
        # 요약 정보
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
        
        # JSON 저장
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / 'analysis_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📄 결과 저장: {results_file}")
        
        print("\n" + "="*70)
        print("✅ 분석 완료!")
        print("="*70)
        
        return self.results


# 사용 예시
if __name__ == '__main__':
    analyzer = IntegratedDISEAnalyzer(
        model_path='checkpoints/best_model.pth',
        fps_extract=5,
        threshold_percent=30,
        min_event_duration=1.0
    )
    
    results = analyzer.analyze_video(
        video_path='test_video.mp4',
        output_dir='analysis_output'
    )
    
    print(f"\n📊 요약:")
    print(f"  - 총 구간: {results['summary']['total_segments']}개")
    print(f"  - 폐색 이벤트: {results['summary']['total_events']}개")
