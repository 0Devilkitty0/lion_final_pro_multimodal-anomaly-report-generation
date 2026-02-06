import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

class VisualRAG:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.feature_bank = None
        self.path_bank = []

    def build_db(self, datamodule):
        """정상 이미지들의 특징을 추출하여 벡터 DB 구축"""
        self.model.eval()
        features_list = []
        self.path_bank = []
        
        # datamodule에서 학습용 데이터 로더 추출
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        
        print(f"📦 특징 추출 시작...")
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Indexing"):
                images = batch["image"].to(self.device)
                paths = batch["image_path"]
                
                # EfficientAD Teacher 모델 활용
                features = self.model.model.teacher(images)
                # avg_features = F.avg_pool2d(features, features.shape[-2:]).view(features.shape[0], -1)
                avg_features = F.adaptive_avg_pool2d(features, (4, 4)).view(features.shape[0], -1)
                
                features_list.append(avg_features.cpu())
                self.path_bank.extend(paths)

        self.feature_bank = torch.cat(features_list, dim=0)
        print(f"✅ DB 구축 완료 ({len(self.path_bank)}개 샘플)")

    def save_db(self, dataset_name, category_name, base_path="results/rag_db"):
        """파일로 저장"""
        os.makedirs(base_path, exist_ok=True)
        # save_path = os.path.join(base_path, f"{dataset_name}_{category_name}.pt")
        save_path = os.path.join(base_path, f"{dataset_name}_{category_name}_4_4_pool.pt")
        torch.save({
            'dataset': dataset_name,
            'category': category_name,
            'feature_bank': self.feature_bank,
            'path_bank': self.path_bank
        }, save_path)
        print(f"💾 DB 저장 완료: {save_path}")

    def load_db(self, dataset_name, category_name, base_path="results/rag_db"):
        """파일에서 로드"""
        # load_path = os.path.join(base_path, f"{dataset_name}_{category_name}.pt")
        load_path = os.path.join(base_path, f"{dataset_name}_{category_name}_4_4_pool.pt")

        if not os.path.exists(load_path):
            return False
        data = torch.load(load_path)
        self.feature_bank = data['feature_bank']
        self.path_bank = data['path_bank']
        return True

    def retrieve(self, test_image_tensor, top_k=5):
        """가장 유사한 정상 이미지 경로 리스트 반환"""
        self.model.eval()
        with torch.no_grad():
            # 1. Feature 추출 및 Global Average Pooling
            test_feat = self.model.model.teacher(test_image_tensor.to(self.device))
            # test_feat = F.avg_pool2d(test_feat, test_feat.shape[-2:]).view(test_feat.shape[0], -1).cpu()
            test_feat = F.adaptive_avg_pool2d(test_feat, (4, 4)).view(test_feat.shape[0], -1).cpu()
            
            # 2. 모든 정상 이미지(feature_bank)와의 거리 계산
            distances = torch.cdist(test_feat, self.feature_bank).squeeze(0) # (N,)
            
            # 3. 거리가 가장 짧은(유사한) 상위 K개 추출
            # distances가 작을수록 유사하므로 largest=False 설정
            topk_values, topk_indices = torch.topk(distances, k=min(top_k, len(self.path_bank)), largest=False)
            
        # 상위 K개의 경로를 리스트로 반환
        return [self.path_bank[idx] for idx in topk_indices.tolist()]
    
    def integrate_with_predictions(self, dataset, category, predictions_root="output/predictions"):
        """기존 predictions.json을 읽어 Top-5 RAG 결과를 추가합니다."""
        from torchvision import transforms
        from PIL import Image
        from pathlib import Path
        import json
        from tqdm import tqdm

        # 1. 기존 예측 결과 경로 설정
        pred_dir = Path(predictions_root) / self.model.__class__.__name__.lower() / dataset / category
        pred_json_path = pred_dir / "predictions.json"
        
        if not pred_json_path.exists():
            print(f"❌ 예측 결과 파일을 찾을 수 없습니다: {pred_json_path}")
            return

        with open(pred_json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        # 2. 이미지 전처리 설정 (EfficientAD 기본 사이즈 256)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # 3. 매칭 시작
        print(f"🔗 [{category}] 예측 결과와 Top-5 정상 이미지 매칭 중...")
        for res in tqdm(predictions, desc="Matching Top-5"):
            img_path = res["image_path"]
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
                # Top-5 경로 리스트 가져오기
                top5_paths = self.retrieve(img_tensor, top_k=5)
                
                # JSON에 리스트 형태로 저장
                res["top5_normal_paths"] = top5_paths
                # (옵션) 가장 유사한 첫 번째는 기존 키 유지 가능
                res["matched_normal_path"] = top5_paths[0] 
                
            except Exception as e:
                print(f"⚠️ {img_path} 처리 중 오류: {e}")
                res["top5_normal_paths"] = []

        # 4. 결과 저장
        # output_path = pred_dir / "predictions_with_rag_top5.json"
        output_path = pred_dir / "predictions_with_rag_top5_4_4_pool.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 통합 완료! Top-5 결과 저장됨: {output_path}")