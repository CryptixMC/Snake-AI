{
  description = "Snake AI — Rust";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in {
        devShells.default = pkgs.mkShell rec {
          name = "snake-ai-rust";

          buildInputs = with pkgs; [
            cargo rustc clippy rust-analyzer rustfmt
            pkg-config
            # macroquad / miniquad — OpenGL + X11
            libGL mesa
            libx11 libxi libxcursor libxrandr libxinerama
            libxkbcommon wayland
            # wgpu — Vulkan backend for AMD
            vulkan-loader
            vulkan-headers
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          # Point wgpu at the RADV ICD that comes with mesa
          VK_ICD_FILENAMES =
            "${pkgs.mesa.drivers}/share/vulkan/icd.d/radeon_icd.x86_64.json";

          shellHook = ''
            echo "Snake AI Rust dev shell"
            rustc --version && cargo --version
          '';
        };
      }
    );
}
