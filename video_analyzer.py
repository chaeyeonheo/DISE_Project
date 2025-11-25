"""
수면 내시경 비디오 분석 파이프라인
- 프레임 추출
- Color-based ROI 검출
- 폐색 영역 분석
- 이상 시점 감지
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import timedelta
import json
from tqdm import tqdm


class AirwayOcclusionAnalyzer:
    """기도 폐색 분석기"""
    
    def __init__(self, fps_extract=5, threshold_percent=30,exclude_first_seconds=2, exclude_last_seconds=3):
        """
        Args:
            fps_extract: 초당 추출할 프레임 수 (예: 5 = 1초에 5프레임)
            threshold_percent: 폐색 기준 (기준 대비 몇 % 감소 시 이상으로 판단)
        """
        self.fps_extract = fps_extract
        self.threshold_percent = threshold_percent
        self.exclude_last_seconds = exclude_last_seconds
        self.exclude_first_seconds = exclude_first_seconds
        self.results = {
            'frames': [],
            'max_area': 0,
            'max_area_frame': 0,
            'occlusion_events': []
        }
    
    def preprocess_frame(self, frame):
        """프레임 전처리 (검정 배경 제거 - 여유있게 크롭)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 🔑 핵심 수정: 안쪽으로 여유 두기 (모서리 제거)
            margin_percent = 0.075  # 10% 안쪽으로
            margin_x = int(w * margin_percent)
            margin_y = int(h * margin_percent)
            
            x = x + margin_x
            y = y + margin_y
            w = w - 2 * margin_x
            h = h - 2 * margin_y
            
            # 범위 체크
            x = max(0, x)
            y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)
            
            return frame[y:y+h, x:x+w], (x, y, w, h)
        
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    
    def extract_roi_color_based(self, frame):
        """Color-based ROI 추출 (기도 내부 어두운 영역)"""
        if frame is None or frame.size == 0:
            return None
        
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
            mask_final = (labels == largest_label).astype('uint8') * 255
            area = stats[largest_label, cv2.CC_STAT_AREA]
            center = centroids[largest_label]
            
            contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                return {
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': tuple(center),
                    'mask': mask_final,
                    'contour': largest_contour
                }
        
        return None
    
    def analyze_video(self, video_path, output_dir=None):
        """
        비디오 분석 메인 함수
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 결과 저장 디렉토리 (None이면 저장 안 함)
        
        Returns:
            results: 분석 결과 딕셔너리
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps and fps > 0 else 0.0

        # 앞/뒤 트림 프레임 수
        exclude_first_frames = int((fps or 0) * max(0, self.exclude_first_seconds))
        exclude_last_frames  = int((fps or 0) * max(0, self.exclude_last_seconds))

        start_frame = min(exclude_first_frames, max(0, total_frames - 1))
        end_frame_exclusive = max(start_frame, total_frames - exclude_last_frames)
        effective_total_frames = max(0, end_frame_exclusive - start_frame)

        print("\n📹 비디오 정보")
        print(f"  - 파일: {video_path.name}")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - 총 프레임: {total_frames}")
        print(f"  - 길이: {duration:.2f}초")
        print(f"  - 앞쪽 제외: {self.exclude_first_seconds}s → {exclude_first_frames}프레임")
        print(f"  - 뒤쪽 제외: {self.exclude_last_seconds}s → {exclude_last_frames}프레임")
        print(f"  - 실제 처리 구간: [{start_frame} .. {end_frame_exclusive-1}]")
        print(f"  - 실제 처리 프레임 수: {effective_total_frames}")

        # 시작 위치로 이동
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 프레임 추출 간격
        frame_interval = int(fps / self.fps_extract) if fps and fps > 0 else 1
        frame_interval = max(frame_interval, 1)

        
        # 출력 디렉토리 설정
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            frames_dir = output_path / "frames"
            mask_dir = output_path / "masks"
            overlay_dir = output_path / "overlays"
            crop_dir = output_path / "crops"

            frames_dir.mkdir(exist_ok=True)
            mask_dir.mkdir(exist_ok=True)
            overlay_dir.mkdir(exist_ok=True)
            crop_dir.mkdir(exist_ok=True)
        
        # 분석 시작
        frame_count = start_frame
        extracted_count = 0
        print(f"\n🔍 프레임 분석 시작...")
        
        # 진행바는 실제 처리 프레임 수 기준으로
        from tqdm import tqdm
        pbar = tqdm(total=(end_frame_exclusive - start_frame), desc="프레임 처리")
                
        while True:
            if frame_count >= end_frame_exclusive:
                break
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 간격에 맞춰 추출
            if (frame_count - start_frame) % frame_interval == 0:
                # 전처리
                preprocessed, bbox = self.preprocess_frame(frame)
                
                # ROI 추출
                roi_info = self.extract_roi_color_based(preprocessed)
                
                # 시간 계산
                timestamp = frame_count / fps if fps > 0 else 0
                time_str = str(timedelta(seconds=int(timestamp)))
                
                frame_result = {
                    'frame_number': int(frame_count),
                    'extracted_index': int(extracted_count),
                    'timestamp': float(timestamp),
                    'time_str': time_str,
                    'preprocessing_bbox': tuple(map(int, bbox))
                }
                
                if roi_info:
                    frame_result.update({
                        'roi_area': float(roi_info['area']),
                        'roi_bbox': tuple(map(int, roi_info['bbox'])),
                        'roi_center': tuple(map(float, roi_info['center']))
                    })
                    
                    # 최대 면적 업데이트
                    if roi_info['area'] > self.results['max_area']:
                        self.results['max_area'] = roi_info['area']
                        self.results['max_area_frame'] = frame_count
                        self.results['max_area_frame_index'] = extracted_count
                    
                    # 프레임 저장 (선택사항)
                    if output_dir:
                        # 1. 원본 프레임
                        frame_filename = f"frame_{extracted_count:06d}.jpg"
                        cv2.imwrite(str(frames_dir / frame_filename), preprocessed)
                        frame_result['saved_path'] = str(frames_dir / frame_filename)
                        
                        # 2. 마스크 이미지 (디버깅용)
                        mask_filename = f"mask_{extracted_count:06d}.png"
                        cv2.imwrite(str(mask_dir / mask_filename), roi_info['mask'])
                        frame_result['mask_path'] = str(mask_dir / mask_filename)
                                                
                        # 3. ROI Crop 이미지
                        x, y, w, h = roi_info['bbox']
                        roi_crop = preprocessed[y:y+h, x:x+w]
                        crop_filename = f"roi_crop_{extracted_count:06d}.jpg"
                        cv2.imwrite(str(crop_dir / crop_filename), roi_crop)
                        frame_result['crop_path'] = str(crop_dir / crop_filename)
                        
                        # 4. Overlay 이미지 (원본 + 반투명 마스크)
                        overlay = preprocessed.copy()
                        mask_colored = np.zeros_like(overlay)
                        mask_colored[roi_info['mask'] > 0] = [255, 255, 0]  # 시안색/청록색 (BGR: 노랑)
                        overlay = cv2.addWeighted(overlay, 0.9, mask_colored, 0.2, 0) # 여기!!! 수정!!! overlay 비율 조절
                        
                        # ROI bbox 그리기 (두께 증가)
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 5)
                        
                        # 중심점 표시
                        center_x, center_y = int(roi_info['center'][0]), int(roi_info['center'][1])
                        cv2.circle(overlay, (center_x, center_y), 10, (0, 0, 255), -1)
                        
                        overlay_filename = f"overlay_{extracted_count:06d}.jpg"
                        cv2.imwrite(str(overlay_dir / overlay_filename), overlay)
                        frame_result['overlay_path'] = str(overlay_dir / overlay_filename)
                else:
                    frame_result['roi_area'] = 0
                
                self.results['frames'].append(frame_result)
                extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # 폐색 이벤트 감지
        self._detect_occlusion_events()
        
        # 메타데이터 추가
        self.results['metadata'] = {
            'video_file': str(video_path),
            'total_frames': total_frames,
            'extracted_frames': extracted_count,
            'fps': fps,
            'duration_seconds': duration,
            'extraction_fps': self.fps_extract,
            'threshold_percent': self.threshold_percent
        }
        
        print(f"\n✅ 분석 완료!")
        print(f"  - 추출된 프레임: {extracted_count}개")
        print(f"  - 최대 ROI 면적: {self.results['max_area']:.0f} px² (프레임 {self.results['max_area_frame']})")
        print(f"  - 폐색 이벤트: {len(self.results['occlusion_events'])}개")
        
        return self.results
    
    def _detect_occlusion_events(self):
        """폐색 이벤트 감지"""
        if self.results['max_area'] == 0:
            return
        
        threshold_area = self.results['max_area'] * (1 - self.threshold_percent / 100)
        
        for frame_data in self.results['frames']:
            if frame_data.get('roi_area', 0) > 0:
                area_reduction = (1 - frame_data['roi_area'] / self.results['max_area']) * 100
                frame_data['area_reduction_percent'] = area_reduction
                
                if frame_data['roi_area'] < threshold_area:
                    event = {
                        'frame_number': frame_data['frame_number'],
                        'extracted_index': frame_data['extracted_index'],
                        'timestamp': frame_data['timestamp'],
                        'time_str': frame_data['time_str'],
                        'roi_area': frame_data['roi_area'],
                        'area_reduction_percent': area_reduction,
                        'severity': self._classify_severity(area_reduction)
                    }
                    
                    # 이미지 경로 추가
                    if 'saved_path' in frame_data:
                        event['frame_path'] = frame_data['saved_path']
                    if 'overlay_path' in frame_data:
                        event['overlay_path'] = frame_data['overlay_path']
                    if 'mask_path' in frame_data:
                        event['mask_path'] = frame_data['mask_path']
                    if 'crop_path' in frame_data:
                        event['crop_path'] = frame_data['crop_path']
                    
                    self.results['occlusion_events'].append(event)
    
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
    
    def save_results(self, output_path):
        """결과를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"📄 결과 저장: {output_path}")


# ========== 테스트 (이미지 시퀀스로 테스트) ==========
if __name__ == "__main__":
    print("=" * 60)
    print("기도 폐색 분석 파이프라인 테스트")
    print("=" * 60)
    
    # 이미지 시퀀스로 간단 테스트
    analyzer = AirwayOcclusionAnalyzer(fps_extract=1, threshold_percent=30)
    
    print("\n✅ 파이프라인 초기화 완료")
    print(f"  - 추출 FPS: {analyzer.fps_extract}")
    print(f"  - 폐색 기준: {analyzer.threshold_percent}%")
