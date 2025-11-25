"""
OTE/Velum/None 분류를 위한 데이터셋 전처리 파이프라인
- video_analyzer.py의 AirwayOcclusionAnalyzer 활용
- 품질 기반 자동 분류
- 프레임 추출 및 레이블링
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys

# 상위 폴더를 Python path에 추가
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# video_analyzer의 AirwayOcclusionAnalyzer import
from video_analyzer import AirwayOcclusionAnalyzer


class DISEDatasetPreprocessor:
    """
    DISE 비디오를 OTE/Velum/None으로 분류하여 데이터셋 생성
    """
    
    def __init__(self, dataset_path='dataset', output_path='processed_dataset'):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # 클래스별 출력 폴더
        for class_name in ['OTE', 'Velum', 'None']:
            (self.output_path / class_name).mkdir(exist_ok=True)
        
        self.all_annotations = []
    
    def analyze_frame_quality(self, frame):
        """
        프레임 품질 분석
        
        Returns:
            quality_score: 0~1 사이의 품질 점수
            metrics: 각 지표별 값
        """
        if frame is None or frame.size == 0:
            return 0.0, {}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 밝기
        brightness = np.mean(gray) / 255.0
        
        # 2. 선명도 (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = min(laplacian.var() / 1000.0, 1.0)
        
        # 3. 대비
        contrast = np.std(gray) / 128.0
        
        # 4. 엣지 밀도
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 5. 색상 분산
        color_std = np.mean([np.std(frame[:,:,i]) for i in range(3)]) / 128.0
        
        # 종합 품질 점수
        quality_score = (
            0.25 * brightness +
            0.25 * sharpness +
            0.20 * contrast +
            0.15 * edge_density +
            0.15 * color_std
        )
        
        metrics = {
            'brightness': brightness,
            'sharpness': sharpness,
            'contrast': contrast,
            'edge_density': edge_density,
            'color_std': color_std,
            'quality_score': quality_score
        }
        
        return quality_score, metrics
    

# preprocess_with_analyzer.py 내부 메서드 수정

    def is_tissue_color(self, frame):
        """
        프레임이 인체 조직(붉은색/분홍색 계열)인지 확인
        """
        # BGR 평균 계산
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        
        # 1. 붉은색이 파란색/초록색보다 우세해야 함 (조직 특성)
        is_red_dominant = (r_mean > g_mean) and (r_mean > b_mean)
        
        # 2. 적절한 채도가 있어야 함 (회색조 노이즈 제외)
        # R과 G/B의 차이가 일정 수준 이상
        color_diff = r_mean - ((g_mean + b_mean) / 2)
        
        return is_red_dominant and (color_diff > 5)

    def classify_frame(self, frame, roi_area, quality_score, video_type,
                    max_roi_area=None, metrics=None):
        """
        [수정 버전 7: 침/거품(Saliva) 저격 패치]
        - 정상 OTE는 매끄럽지만, 침/거품은 '자글자글'하다는 점을 이용
        - Edge Score가 '너무 높으면' 노이즈(거품)로 간주하여 제거
        """
        
        # 공통 기초 통계량
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # 1. 절대 방출 기준
        if mean_brightness < 5: return 'None', 0.99
        if mean_brightness > 250: return 'None', 0.99

        # ---------------------------------------------------------
        # 2. [OTE 전용] 침/거품 저격 & 구조 검사
        # ---------------------------------------------------------
        if video_type == 'OTE':
            # (1) Canny Edge 계산
            # 민감도(20, 80) 유지하되, 결과를 해석하는 방법을 바꿈
            edges = cv2.Canny(gray, 20, 80)
            edge_score = np.mean(edges)
            
            # (2) [신규] 침/거품 함정 (Saliva Trap) 🕸️
            # 거품은 테두리가 많아서 Edge Score가 비정상적으로 높게 나옴 (보통 10~15 이상)
            # 반면 정상 기도는 매끄러워서 보통 3~8 사이가 나옴.
            # ROI가 초대형(>3000)이 아닌데 Edge가 너무 많으면 -> 100% 거품임
            if edge_score > 12.0 and roi_area < 3000:
                return 'None', 0.95

            # (3) [신규] 반사광(Specular Highlight) 저격
            # 침방울은 빛을 반사해 국소적으로 엄청 밝고 선명함
            # 선명도(Sharpness)가 비정상적으로 높은데 ROI는 작다? -> 침방울
            sharpness = metrics.get('sharpness', 0) if metrics else 0
            if sharpness > 0.05 and roi_area < 1000:  # 0.05는 꽤 높은 수치
                return 'None', 0.90

            # (4) 구멍(ROI) 우선 구제 (V6 로직 유지)
            # 거품이 있어도 진짜 기도가 뻥 뚫려있으면(>1000) OTE로 인정
            if roi_area > 1000:
                return 'OTE', 0.95

            # (5) 최소 엣지 점수 심사 (V6 유지)
            # 너무 맹탕(안개)인 것만 제거
            if edge_score < 0.5:
                if roi_area < 200: return 'None', 0.95
                
            # (6) 조직 색상 검사
            is_tissue = self.is_tissue_color(frame)
            if not is_tissue and roi_area < 500:
                return 'None', 0.85

            return 'OTE', 0.85

        # ---------------------------------------------------------
        # 3. [Velum 전용] 기존 유지
        # ---------------------------------------------------------
        else:
            contrast = metrics.get('contrast', 0) if metrics else 0
            if contrast < 0.02: return 'None', 0.95
            
            is_tissue = self.is_tissue_color(frame)
            sharpness = metrics.get('sharpness', 0) if metrics else 0
            
            if is_tissue and sharpness < 0.001:
                return 'None', 0.90

            if roi_area < 50 and quality_score < 0.10:
                return 'None', 0.85

            confidence = 0.8
            if roi_area > 500: confidence += 0.15
            if is_tissue: confidence += 0.15
            
            if roi_area < 50 and not is_tissue and mean_brightness < 40:
                return 'None', 0.85
                
            return 'Velum', min(confidence, 1.0)

    def process_video_with_analyzer(self, video_path, video_type):
        """
        video_analyzer를 사용하여 비디오 처리
        
        Args:
            video_path: 비디오 파일 경로
            video_type: 'OTE' or 'Velum'
        """
        print(f"\n{'='*70}")
        print(f"Processing: {video_path.name} (Type: {video_type})")
        print(f"{'='*70}")
        
        # AirwayOcclusionAnalyzer 생성
        analyzer = AirwayOcclusionAnalyzer(
            fps_extract=5,  # 초당 5프레임 추출
            threshold_percent=30,
            exclude_first_seconds=2,
            exclude_last_seconds=3
        )
        
        # 분석 실행
        results = analyzer.analyze_video(str(video_path), output_dir=None)
        
        # ⭐ 최대 ROI 면적 (전체 프레임 중)
        max_roi_area = results['max_area']
        
        print(f"\n📊 비디오 ROI 통계:")
        print(f"  - 최대 ROI 면적: {max_roi_area:.0f} px²")
        
        # 이제 다시 프레임별로 분류
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        video_name = video_path.stem
        frame_count = 0
        saved_count = {'OTE': 0, 'Velum': 0, 'None': 0}
        
        print(f"\nClassifying and saving frames...")
        
        for frame_data in tqdm(results['frames'], desc="Frames"):
            frame_number = frame_data['frame_number']
            roi_area = frame_data.get('roi_area', 0)
            
            # 프레임 읽기
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 프레임 전처리 (analyzer와 동일)
            preprocessed, bbox = analyzer.preprocess_frame(frame)
            
            # 품질 분석
            quality_score, metrics = self.analyze_frame_quality(preprocessed)
            
            # 분류
            label, confidence = self.classify_frame(
                preprocessed, roi_area, quality_score,
                video_type, max_roi_area, metrics
            )
            
            # 프레임 저장
            output_folder = self.output_path / label
            frame_filename = f"{video_name}_{label}_frame_{saved_count[label]:06d}.jpg"
            frame_path = output_folder / frame_filename
            
            cv2.imwrite(str(frame_path), preprocessed)
            
            # 어노테이션 저장
            annotation = {
                'filename': frame_filename,
                'label': label,
                'video_name': video_name,
                'video_type': video_type,
                'frame_number': frame_number,
                'timestamp': frame_data['timestamp'],
                'roi_area': roi_area,
                'quality_score': quality_score,
                'confidence': confidence,
                'metrics': metrics,
                'path': str(frame_path)
            }
            
            self.all_annotations.append(annotation)
            saved_count[label] += 1
        
        cap.release()
        
        # 통계 출력
        print(f"\n✓ {video_path.name} 처리 완료")
        print(f"  - OTE: {saved_count['OTE']} frames")
        print(f"  - Velum: {saved_count['Velum']} frames")
        print(f"  - None: {saved_count['None']} frames")
        
        return saved_count
    
    def process_all_videos(self):
        """모든 비디오 처리"""
        total_stats = {'OTE': 0, 'Velum': 0, 'None': 0}
        
        # Velum 비디오 처리
        velum_path = self.dataset_path / 'Velum'
        if velum_path.exists():
            velum_videos = list(velum_path.glob('*.mp4'))
            print(f"\n📹 Found {len(velum_videos)} Velum videos")
            
            for video_file in velum_videos:
                stats = self.process_video_with_analyzer(video_file, 'Velum')
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        # OTE 비디오 처리
        ote_path = self.dataset_path / 'OTE'
        if ote_path.exists():
            ote_videos = list(ote_path.glob('*.mp4'))
            print(f"\n📹 Found {len(ote_videos)} OTE videos")
            
            for video_file in ote_videos:
                stats = self.process_video_with_analyzer(video_file, 'OTE')
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        return total_stats
    
    def save_annotations(self):
        """어노테이션 파일 저장"""
        annotation_file = self.output_path / 'annotations.json'
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Annotations saved: {annotation_file}")
        
        # 통계 파일 생성
        stats = {
            'total_frames': len(self.all_annotations),
            'class_distribution': {},
            'video_types': {}
        }
        
        for ann in self.all_annotations:
            label = ann['label']
            video_type = ann['video_type']
            
            stats['class_distribution'][label] = stats['class_distribution'].get(label, 0) + 1
            stats['video_types'][video_type] = stats['video_types'].get(video_type, 0) + 1
        
        stats_file = self.output_path / 'dataset_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Statistics saved: {stats_file}")
        
        return stats
    
    def run(self):
        """전체 파이프라인 실행"""
        print("\n" + "="*70)
        print("DISE Dataset Preprocessing Pipeline")
        print("="*70)
        
        # 모든 비디오 처리
        total_stats = self.process_all_videos()
        
        # 어노테이션 저장
        stats = self.save_annotations()
        
        # 최종 통계
        print("\n" + "="*70)
        print("=== Final Statistics ===")
        print("="*70)
        print(f"Total frames: {stats['total_frames']}")
        print(f"\nClass distribution:")
        for label, count in stats['class_distribution'].items():
            percentage = count / stats['total_frames'] * 100
            print(f"  {label}: {count} frames ({percentage:.1f}%)")
        
        print(f"\nVideo types:")
        for vtype, count in stats['video_types'].items():
            print(f"  {vtype}: {count} frames")
        
        print("\n" + "="*70)
        print("✅ Preprocessing completed!")
        print("="*70)
        
        return stats


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DISE Video Dataset Preprocessing'
    )
    parser.add_argument('--dataset', type=str, default='D:/chaeyeon/대학원/3학기/융프/data/DISE_DATA(AIHub)/little',
                       help='Dataset directory containing OTE/ and Velum/ folders')
    parser.add_argument('--output', type=str, default='processed_dataset',
                       help='Output directory for processed frames')
    
    args = parser.parse_args()
    
    # 전처리 실행
    preprocessor = DISEDatasetPreprocessor(
        dataset_path=args.dataset,
        output_path=args.output
    )
    
    stats = preprocessor.run()
    
    print(f"\n💡 Next steps:")
    print(f"1. Check the frames in {args.output}/[OTE|Velum|None]/")
    print(f"2. Review dataset_stats.json for class distribution")
    print(f"3. Run: python train.py")


if __name__ == '__main__':
    main()
