from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import gc
import inspect

def _import_model(model_name: str):
    """EfficientAD 모델 클래스 임포트 (버전 호환성 유지)"""
    EfficientAD = None
    # 대소문자 후보군 순회
    for cand in ("EfficientAd", "EfficientAD", "Efficientad"):
        try:
            mod = __import__("anomalib.models", fromlist=[cand])
            EfficientAD = getattr(mod, cand)
            break
        except Exception:
            continue
    if EfficientAD is None:
        raise ImportError("EfficientAD 모델을 찾을 수 없습니다. anomalib 설치를 확인하세요.")
    return EfficientAD

def _find_ckpt(output_dir: Path) -> Path:
    """학습 결과물 폴더에서 가장 최근 체크포인트(.ckpt) 탐색"""
    candidates = [output_dir / "weights" / "lightning" / "model.ckpt"]
    for c in candidates:
        if c.exists(): return c
    ckpts = list(output_dir.rglob("*.ckpt"))
    if not ckpts: raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {output_dir}")
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]

def run_one_category(category: str, args: argparse.Namespace) -> Path | None:
    """개별 카테고리(상품) 학습 수행"""
    try:
        # sys.path 설정 후 src 모듈 임포트
        from src.datasets.mmad_index_csv import load_mmad_index_csv, filter_by_category, split_good_train_test
        from src.datasets.anomalib_folder_builder import build_anomalib_folder_dataset
        from src.utils.log import setup_logger
        
        logger = setup_logger(name="TrainAnomalib", log_prefix="train_anomalib")

        # 1. 데이터 로드 (MMAD_index.csv 대소문자 주의)
        records = load_mmad_index_csv(args.index_csv, data_root=args.data_root)
        cat_records = filter_by_category(records, category)
        
        if not cat_records:
            logger.warning(f"[{category}] 데이터가 없습니다. 건너뜁니다.")
            return None

        # 2. Anomalib 전용 Folder Dataset 빌드
        work_dir = Path(args.work_dir)
        train_goods, test_records = split_good_train_test(cat_records, train_ratio=args.train_ratio, seed=args.seed)
        
        built = build_anomalib_folder_dataset(
            train_goods=train_goods,
            test_records=test_records,
            out_root=work_dir,
            category=category,
            copy_files=bool(args.copy_files),
        )
        cat_root = Path(built.root) / built.category

        # 3. Anomalib 컴포넌트 준비
        from anomalib.data import Folder
        from anomalib.engine import Engine

        model = _import_model("efficientad")()
        
        datamodule = Folder(
            name=category,
            root=str(cat_root),
            normal_dir="train/good",
            train_batch_size=int(args.train_batch_size),
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )

        # 결과 저장 경로 (outputs_anomalib 하위)
        out_dir = Path(args.output_dir) / "efficientad" / category
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4. 학습 엔진 설정
        engine = Engine(
            default_root_dir=str(out_dir),
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=min(args.max_epochs, 50), # 로그 폭주 방지
            num_sanity_val_steps=0,
        )

        logger.info(f"=== [{category}] 학습 시작 (Total Categories: {args.total_count}) ===")
        engine.fit(model=model, datamodule=datamodule)

        ckpt = _find_ckpt(out_dir)
        
        # GPU 메모리 해제 (중요: 클래스 순회 시 메모리 누수 방지)
        del engine, model, datamodule
        torch.cuda.empty_cache()
        gc.collect()
        
        return ckpt

    except Exception as e:
        print(f"Error occurred in category '{category}': {e}")
        return None

def main() -> None:
    ap = argparse.ArgumentParser(description="EfficientAD All-in-One Training Script")
    ap.add_argument("--project-root", type=str, default=str(Path.cwd()), help="프로젝트 루트 경로")
    ap.add_argument("--category", type=str, default="all", help="특정 카테고리 혹은 'all'")
    ap.add_argument("--max-epochs", type=int, default=700)
    ap.add_argument("--train-batch-size", type=int, default=1)
    ap.add_argument("--copy-files", action="store_true", help="Windows 환경이거나 심볼릭 링크 문제 시 사용")
    args = ap.parse_args()

    # --- 경로 설정 규칙 (대소문자 엄격 적용) ---
    proj_root = Path(args.project_root).resolve()
    
    # PROJECT_ROOT/dataset/MMAD/MMAD_index.csv
    args.index_csv = str(proj_root / "dataset" / "MMAD" / "MMAD_index.csv")
    
    # PROJECT_ROOT/dataset
    args.data_root = str(proj_root / "dataset" / "MMAD")
    
    args.work_dir = str(proj_root / "data_anomalib")
    args.output_dir = str(proj_root / "outputs_anomalib")
    args.train_ratio = 0.9
    args.seed = 42
    args.eval_batch_size = 1
    args.num_workers = 4

    # src 폴더를 찾기 위해 sys.path 추가
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    # 파일 존재 확인
    if not Path(args.index_csv).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.index_csv}")
        print("경로와 대소문자를 다시 확인해주세요.")
        return

    print(f"✅ Project Root: {proj_root}")
    print(f"✅ CSV Path: {args.index_csv}")

    # 모든 카테고리 목록 추출
    from src.datasets.mmad_index_csv import load_mmad_index_csv
    all_records = load_mmad_index_csv(args.index_csv, data_root=args.data_root)
    categories = sorted({r.category for r in all_records})
    args.total_count = len(categories)

    if args.category.lower() == "all":
        print(f"🚀 총 {args.total_count}개의 카테고리 학습을 순차적으로 진행합니다.")
        for idx, cat in enumerate(categories, 1):
            print(f"\n({idx}/{args.total_count}) Working on: {cat}")
            run_one_category(cat, args)
    else:
        run_one_category(args.category, args)

if __name__ == "__main__":
    main()