import os
import subprocess
from pathlib import Path

def run_command(command):
    """명령어를 실행하고 결과를 실시간으로 출력합니다."""
    print(f"\n🏃 실행 중: {command}")
    try:
        env = os.environ.copy()
        # 자동 약관 동의 환경변수 추가
        env["CONDA_PLUGINS_AUTO_ACCEPT_TOS"] = "yes"
        env["MPLBACKEND"] = "Agg"
        
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, env=env, executable='/bin/bash'
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def main():
    print("========================================================================")
    print("🌟 Anomalib + AI Models 통합 설치 (ToS Fix 적용)")
    print("========================================================================")

    CONDA_BASE = "/content/conda"
    CONDA_BIN = f"{CONDA_BASE}/bin/conda"
    ENV_PATH = f"{CONDA_BASE}/envs/anomaly_report"
    PY = f"{ENV_PATH}/bin/python"
    UV = f"{CONDA_BASE}/bin/uv"

    # 1. 기초 환경 (Miniconda) 설치
    if not os.path.exists(CONDA_BASE):
        print("\n1️⃣ Miniconda 설치 중...")
        run_command("wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh")
        run_command(f"bash /tmp/miniconda.sh -b -p {CONDA_BASE}")
        run_command(f"{CONDA_BASE}/bin/pip install uv -q")

    # 2. 약관 동의 및 가상환경 생성
    if not os.path.exists(ENV_PATH):
        print("\n2️⃣ 약관 동의 및 환경 생성 중...")
        # 약관 동의 명령어를 명시적으로 먼저 실행
        run_command(f"{CONDA_BIN} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main")
        run_command(f"{CONDA_BIN} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r")
        # 환경 생성
        run_command(f"{CONDA_BIN} create -n anomaly_report python=3.10 -y -q")
    else:
        print("\n2️⃣ anomaly_report 환경 이미 존재")

    # 3. 핵심 엔진 설치 (PyTorch + CUDA 11.8)
    if os.path.exists(PY):
        print("\n3️⃣ PyTorch 2.1.2 + cu118 설치...")
        torch_install = (
            f"{UV} pip install 'torch==2.1.2' 'torchvision==0.16.2' 'numpy==1.26.4' "
            f"--index-url https://download.pytorch.org/whl/cu118 --python {PY} -q"
        )
        run_command(torch_install)

        # 4. Anomalib 및 모델 의존성 라이브러리 설치
        print("\n4️⃣ Anomalib 및 모델 의존성 설치...")
        libs_to_install = [
            "anomalib==1.1.0", "lightning==2.1.4", "torchmetrics==1.2.1",
            "open_clip_torch", "FrEIA", "einops", "timm", "kornia", 
            "imgaug", "omegaconf", "rich", "opencv-python-headless==4.10.0.84",
            "scikit-learn==1.3.2", "scikit-image==0.21.0", "seaborn==0.13.2",
            "pandas==2.2.2", "matplotlib==3.8.4", "pyyaml==6.0.2", "tqdm==4.66.5"
        ]
        libs_str = " ".join([f"'{lib}'" for lib in libs_to_install])
        run_command(f"{UV} pip install {libs_str} --python {PY} -q")
    else:
        print(f"❌ 에러: 파이썬 경로를 찾을 수 없습니다: {PY}")

    # 5. 최종 확인
    print("\n" + "="*72)
    print("🔍 최종 검증 결과")
    print("="*72)
    if os.path.exists(PY):
        verify_script = """
import numpy as np
import torch
import anomalib
try:
    import open_clip
    import FrEIA
    clip_ok = True
except:
    clip_ok = False
print(f'✅ NumPy:    {np.__version__} (1.26.4)')
print(f'✅ PyTorch:  {torch.__version__} (2.1.2+cu118)')
print(f'✅ GPU:      {torch.cuda.is_available()}')
print(f'✅ Anomalib: {anomalib.__version__}')
print(f'✅ Models:   {"Ready (Clip, FrEIA installed)" if clip_ok else "Missing extras"}')
"""
        with open("/tmp/verify.py", "w") as f: f.write(verify_script)
        run_command(f"{PY} /tmp/verify.py")
    else:
        print("❌ 환경 생성 실패. 로그를 확인하세요.")
    print("="*72)

if __name__ == "__main__":
    main()