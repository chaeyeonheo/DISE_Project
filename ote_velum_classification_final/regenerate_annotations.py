import json
from pathlib import Path
from tqdm import tqdm
import os

def regenerate_annotations(dataset_root='processed_dataset'):
    root_path = Path(dataset_root)
    annotation_file = root_path / 'annotations.json'
    
    print(f"🔄 Regenerating annotations based on folder structure in: {root_path}")
    
    # 1. 기존 어노테이션 백업 (메타데이터 보존용)
    old_metadata = {}
    if annotation_file.exists():
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                # 파일명을 키(Key)로 해서 메타데이터 저장
                for item in old_data:
                    old_metadata[item['filename']] = item
            print(f"✅ Loaded {len(old_metadata)} existing annotations for metadata preservation.")
        except Exception as e:
            print(f"⚠️ Could not load existing annotations: {e}")
            print("   Generating new metadata from scratch.")
    else:
        print("ℹ️ No existing annotations found. Creating new ones.")

    new_annotations = []
    stats = {'OTE': 0, 'Velum': 0, 'None': 0}
    
    # 2. 현재 폴더 구조 스캔
    # 정의된 클래스 폴더들
    classes = ['OTE', 'Velum', 'None']
    
    for class_name in classes:
        class_dir = root_path / class_name
        if not class_dir.exists():
            print(f"⚠️ Warning: Folder {class_name} does not exist.")
            continue
            
        # 이미지 파일 찾기
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        print(f"📂 Scanning {class_name}: Found {len(images)} images")
        
        for img_path in tqdm(images, desc=f"Processing {class_name}"):
            filename = img_path.name
            
            # 새 어노테이션 항목 생성
            entry = {
                'filename': filename,
                'label': class_name,  # ⭐ 현재 폴더 이름이 곧 정답 라벨!
                'path': str(img_path).replace(os.sep, '/')  # 윈도우 경로 호환성
            }
            
            # 기존 메타데이터가 있으면 복구 (ROI 면적, 타임스탬프 등)
            if filename in old_metadata:
                old_entry = old_metadata[filename]
                # 기존 정보 복사하되, 핵심 정보는 현재 상태로 덮어쓰기
                entry.update(old_entry)
                entry['label'] = class_name  # 라벨은 무조건 현재 폴더 기준
                entry['path'] = str(img_path).replace(os.sep, '/')
            else:
                # 기존 정보가 없으면 (이름을 바꿨거나 새로 넣은 파일)
                # 파일명에서 정보 유추 시도 (형식: VideoName_Label_frame_XXXXXX.jpg)
                try:
                    parts = img_path.stem.split('_frame_')
                    if len(parts) == 2:
                        entry['video_name'] = parts[0]
                        entry['frame_number'] = int(parts[1])
                except:
                    pass
            
            new_annotations.append(entry)
            stats[class_name] += 1

    # 3. 저장
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(new_annotations, f, indent=2, ensure_ascii=False)
    
    # 통계 저장
    stats_file = root_path / 'dataset_stats.json'
    final_stats = {
        'total_frames': len(new_annotations),
        'class_distribution': stats
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    print(f"\n✨ Successfully regenerated 'annotations.json'!")
    print(f"📍 Total images: {len(new_annotations)}")
    print(f"📊 Distribution: {stats}")
    
    return new_annotations

if __name__ == "__main__":
    regenerate_annotations()