"""테스트 데이터 개수 확인 스크립트"""
import json
from pathlib import Path
import random

# 어노테이션 로드
annotation_file = Path('processed_dataset/annotations.json')
with open(annotation_file, 'r', encoding='utf-8') as f:
    all_annotations = json.load(f)

# 레이블별로 그룹화
label_groups = {'OTE': [], 'Velum': [], 'None': []}
for ann in all_annotations:
    label_groups[ann['label']].append(ann)

print("\n=== 전체 데이터 통계 ===")
for label, anns in label_groups.items():
    print(f"{label}: {len(anns)} frames")
print(f"Total: {len(all_annotations)} frames")

# 각 클래스별로 train/val/test 분할
val_split = 0.15
test_split = 0.15

train_anns, val_anns, test_anns = [], [], []

for label, anns in label_groups.items():
    # 비디오 이름으로 그룹화
    video_groups = {}
    for ann in anns:
        video_name = ann['video_name']
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(ann)
    
    video_names = list(video_groups.keys())
    
    # 비디오 단위로 분할 (random_state=42로 고정)
    random.seed(42)
    random.shuffle(video_names)
    
    total_videos = len(video_names)
    test_val_size = int(total_videos * (val_split + test_split))
    test_size = int(total_videos * test_split)
    
    train_videos = video_names[:-test_val_size]
    temp_videos = video_names[-test_val_size:]
    
    val_videos = temp_videos[:-test_size]
    test_videos = temp_videos[-test_size:]
    
    # 각 그룹에 프레임 추가
    for video in train_videos:
        train_anns.extend(video_groups[video])
    for video in val_videos:
        val_anns.extend(video_groups[video])
    for video in test_videos:
        test_anns.extend(video_groups[video])

print(f"\n=== 데이터 분할 결과 ===")
print(f"Train: {len(train_anns)} frames ({len(train_anns)/len(all_annotations)*100:.1f}%)")
print(f"Val: {len(val_anns)} frames ({len(val_anns)/len(all_annotations)*100:.1f}%)")
print(f"Test: {len(test_anns)} frames ({len(test_anns)/len(all_annotations)*100:.1f}%)")

# 클래스별 테스트 데이터 개수
print(f"\n=== Test 데이터 클래스별 분포 ===")
test_label_counts = {'OTE': 0, 'Velum': 0, 'None': 0}
for ann in test_anns:
    test_label_counts[ann['label']] += 1

for label, count in test_label_counts.items():
    print(f"{label}: {count} frames")

