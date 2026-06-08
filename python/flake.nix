{
  description = "Snake AI - Python + ROCm (AMD RX 6800 XT)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            rocmSupport = true;
          };
        };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          torch
          numpy
          pygame
        ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "snake-ai-python";

          packages = [
            pythonEnv
            pkgs.rocmPackages.rocminfo
            pkgs.rocmPackages.rocm-runtime
          ];

          env = {
            HSA_OVERRIDE_GFX_VERSION = "10.3.0";
            ROCR_VISIBLE_DEVICES = "0";
          };

          shellHook = ''
            echo "Snake AI dev shell — ROCm / RX 6800 XT"
            echo "Checking GPU access..."
            rocminfo 2>/dev/null | grep -A2 "Agent 2" | head -6 || echo "  rocminfo failed — check render/video groups"
            echo ""
            python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  ROCm/CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
"
          '';
        };
      }
    );
}
