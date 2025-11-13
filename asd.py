import os
import glob
import random
import shutil

# 루트 디렉토리 설정
root_dir = "./data/database"
save_root = "./data/split"

# 결과 저장 경로 생성
os.makedirs(save_root, exist_ok=True)

# 재현 가능성 보장을 위해 seed 고정
random.seed(42)

# 각 하위 폴더별로 처리
for folder in sorted(glob.glob(os.path.join(root_dir, "*/high"))):
    folder_name = os.path.basename(os.path.dirname(folder))
    print(f"\n📁 폴더 처리 중: {folder_name}")

    # 해당 폴더의 파일 목록 수집
    files = glob.glob(os.path.join(folder, "*"))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("⚠️ 파일이 없습니다. 건너뜁니다.")
        continue

    random.shuffle(files)

    # 분할 비율
    split_ratio = 0.9
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # 폴더별 저장 경로 생성
    train_dir = os.path.join(save_root, "train", folder_name, "high")
    val_dir = os.path.join(save_root, "val", folder_name, "high")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 파일 복사
    for f in train_files:
        shutil.copy(f, train_dir)
    for f in val_files:
        shutil.copy(f, val_dir)

    print(
        f"✅ {folder_name}: Train {len(train_files)}개, Val {len(val_files)}개 분할 완료"
    )

print("\n🎯 모든 폴더 분할 완료!")
