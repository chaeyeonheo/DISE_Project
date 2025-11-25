import cv2
import os
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

class VideoPreprocessor:
    def __init__(self, dataset_path='dataset', output_path='processed_dataset'):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # 각 클래스별 출력 폴더 생성
        for class_name in ['OTE', 'Velum', 'None']:
            (self.output_path / class_name).mkdir(exist_ok=True)
    
    def get_video_info(self, video_path):
        """비디오 정보 추출"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        return fps, total_frames, duration
    
    def extract_frames(self, video_path, start_sec, end_sec, output_folder, label):
        """지정된 구간의 프레임 추출"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        video_name = Path(video_path).stem
        frame_count = 0
        extracted_frames = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 저장
            frame_filename = f"{video_name}_frame_{frame_count:04d}.jpg"
            frame_path = output_folder / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            extracted_frames.append({
                'filename': frame_filename,
                'label': label,
                'video_name': video_name,
                'frame_number': frame_count
            })
            
            frame_count += 1
        
        cap.release()
        return extracted_frames
    
    def process_velum_videos(self):
        """Velum 비디오 처리 (앞뒤 3초 제거)"""
        velum_path = self.dataset_path / 'Velum'
        video_files = list(velum_path.glob('*.mp4'))
        
        print(f"\n=== Processing Velum videos ===")
        print(f"Found {len(video_files)} videos")
        
        all_frames = []
        none_frames = []
        
        for video_file in tqdm(video_files, desc="Velum videos"):
            fps, total_frames, duration = self.get_video_info(video_file)
            
            if duration <= 6:
                print(f"Warning: {video_file.name} is too short ({duration:.2f}s)")
                continue
            
            # Velum 레이블: 앞뒤 3초 제거
            velum_frames = self.extract_frames(
                video_file, 
                start_sec=3, 
                end_sec=duration-3,
                output_folder=self.output_path / 'Velum',
                label='Velum'
            )
            all_frames.extend(velum_frames)
            
            # None 레이블: 앞 3초
            none_start = self.extract_frames(
                video_file,
                start_sec=0,
                end_sec=3,
                output_folder=self.output_path / 'None',
                label='None'
            )
            none_frames.extend(none_start)
            
            # None 레이블: 뒤 3초
            none_end = self.extract_frames(
                video_file,
                start_sec=duration-3,
                end_sec=duration,
                output_folder=self.output_path / 'None',
                label='None'
            )
            none_frames.extend(none_end)
        
        return all_frames, none_frames
    
    def process_ote_videos(self):
        """OTE 비디오 처리 (앞 9초, 뒤 3초 제거)"""
        ote_path = self.dataset_path / 'OTE'
        video_files = list(ote_path.glob('*.mp4'))
        
        print(f"\n=== Processing OTE videos ===")
        print(f"Found {len(video_files)} videos")
        
        all_frames = []
        none_frames = []
        
        for video_file in tqdm(video_files, desc="OTE videos"):
            fps, total_frames, duration = self.get_video_info(video_file)
            
            if duration <= 12:
                print(f"Warning: {video_file.name} is too short ({duration:.2f}s)")
                continue
            
            # OTE 레이블: 앞 9초, 뒤 3초 제거 (Velum 구간 6초 + 기본 3초)
            ote_frames = self.extract_frames(
                video_file,
                start_sec=9,
                end_sec=duration-3,
                output_folder=self.output_path / 'OTE',
                label='OTE'
            )
            all_frames.extend(ote_frames)
            
            # None 레이블: 앞 3초
            none_start = self.extract_frames(
                video_file,
                start_sec=0,
                end_sec=3,
                output_folder=self.output_path / 'None',
                label='None'
            )
            none_frames.extend(none_start)
            
            # None 레이블: 뒤 3초
            none_end = self.extract_frames(
                video_file,
                start_sec=duration-3,
                end_sec=duration,
                output_folder=self.output_path / 'None',
                label='None'
            )
            none_frames.extend(none_end)
        
        return all_frames, none_frames
    
    def create_annotation_file(self, all_annotations):
        """어노테이션 파일 생성"""
        annotation_file = self.output_path / 'annotations.json'
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnnotation file saved: {annotation_file}")
        return annotation_file
    
    def run(self):
        """전체 전처리 실행"""
        print("Starting video preprocessing...")
        
        # Velum 비디오 처리
        velum_frames, velum_none = self.process_velum_videos()
        
        # OTE 비디오 처리
        ote_frames, ote_none = self.process_ote_videos()
        
        # 모든 프레임 수집
        all_none = velum_none + ote_none
        all_annotations = velum_frames + ote_frames + all_none
        
        # 통계 출력
        print("\n=== Preprocessing Summary ===")
        print(f"Total Velum frames: {len(velum_frames)}")
        print(f"Total OTE frames: {len(ote_frames)}")
        print(f"Total None frames: {len(all_none)}")
        print(f"Total frames: {len(all_annotations)}")
        
        # 어노테이션 파일 생성
        self.create_annotation_file(all_annotations)
        
        return all_annotations


if __name__ == '__main__':
    preprocessor = VideoPreprocessor(
        dataset_path='dataset',
        output_path='processed_dataset'
    )
    
    annotations = preprocessor.run()
    
    print("\nPreprocessing completed!")
