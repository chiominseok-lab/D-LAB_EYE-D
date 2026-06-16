"""
EXE 빌드 스크립트 (CPU 전용 최적화 버전)
실행: python build_exe.py

빌드 전 준비:
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

import subprocess
import sys
import os


def check_cpu_only_torch():
    """CPU-only PyTorch 설치 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            print("⚠  경고: CUDA 버전 PyTorch가 감지되었습니다.")
            print("   EXE 크기를 줄이려면 CPU-only 버전으로 교체하세요:")
            print()
            print("   pip uninstall torch torchvision -y")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print()
            ans = input("   그래도 계속 빌드하시겠습니까? (y/N): ").strip().lower()
            if ans != "y":
                print("빌드를 취소했습니다.")
                sys.exit(0)
        else:
            print("✅ CPU-only PyTorch 확인됨")
    except ImportError:
        print("❌ PyTorch가 설치되어 있지 않습니다. 먼저 설치하세요.")
        sys.exit(1)


def main():
    # PyInstaller 설치 확인
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    check_cpu_only_torch()

    script = os.path.join(os.path.dirname(__file__), "app.py")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",            # onefile 대신 onedir (실행 속도 빠름, 크기도 유리)
        "--windowed",          # 콘솔 창 없음
        "--name", "FundusClassifier",
        "--clean",
        "--noupx",             # PyTorch와 UPX 충돌 방지

        # ── 필요한 hidden-import만 명시 ──────────────────────────
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "torchvision.ops",
        "--hidden-import", "torchvision.ops.stochastic_depth",
        "--hidden-import", "torch",
        "--hidden-import", "torch.nn",
        "--hidden-import", "torch.nn.functional",
        "--hidden-import", "webview",
        "--hidden-import", "webview.platforms.winforms",
        "--collect-all", "webview",

        # ── torchvision은 data/binary만 수집 (collect-all 제거) ──
        "--collect-data", "torchvision",
        "--collect-binaries", "torchvision",

        # ── 가중치 파일 번들링 ───────────────────────────────────
        "--add-data", "fold1_best.pth;.",
        "--add-data", "fold2_best.pth;.",
        "--add-data", "fold3_best.pth;.",

        # ── 안전하게 제외 가능한 모듈만 ─────────────────────────
        "--exclude-module", "torch.utils.tensorboard",
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "numpy.testing",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "notebook",
        "--exclude-module", "scipy",
        "--exclude-module", "pandas",
        "--exclude-module", "sklearn",
        "--exclude-module", "sqlalchemy",
        "--exclude-module", "cryptography",

        script,
    ]

    print("=" * 60)
    print("EXE 빌드 시작... (CPU 전용 최적화)")
    print("⚠  약 5~15분 소요될 수 있습니다.")
    print("=" * 60)

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        dist_path = os.path.join(os.path.dirname(__file__), "dist", "FundusClassifier")
        print("\n✅ 빌드 완료!")
        print(f"폴더 경로: {dist_path}")
        print("\n배포 시: dist/FundusClassifier/ 폴더 전체를 zip으로 묶어 전달하세요.")
    else:
        print("\n❌ 빌드 실패. 위의 오류 메시지를 확인하세요.")


if __name__ == "__main__":
    main()
