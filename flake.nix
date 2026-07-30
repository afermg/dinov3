{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nixpkgs.url = "github:NixOS/nixpkgs/efcb904a6c674d1d3717b06b89b54d65104d4ea7";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        libList = [
          pkgs.stdenv.cc.cc
          pkgs.stdenv.cc
          pkgs.libGL
          pkgs.gcc
          #pkgs.gcc.cc.lib
          pkgs.glib
          pkgs.libz
          pkgs.glibc
          #pkgs.glibc.dev
        ];
        modelPackages = {
          dinov3 = pkgs.python3.pkgs.callPackage ./nix/dinov3.nix {};
        };
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          (inputs.nahual-flake.packages.${system}.nahual)
          modelPackages.dinov3
        ]);
        runServer = pkgs.writeScriptBin "nahual-dinov3" ''
          #!${pkgs.bash}/bin/bash
          exec ${python_with_pkgs}/bin/python ${self}/server.py "''${1:-tcp://0.0.0.0:5555}"
        '';
        serverApp = {
          type = "app";
          program = "${runServer}/bin/nahual-dinov3";
        };
      in
        with pkgs; rec {
          apps.default = serverApp;
          packages =
            modelPackages
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "dinov3";
                title = "Nahual DINOv3";
                description = "DINOv3 feature extraction served through Nahual";
                source = "https://github.com/afermg/dinov3";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runServer;
                entrypoint = serverApp.program;
              };
            };
          devShells = {
            default = let
              python_with_pkgs = (
                python3.withPackages (pp: [
                  (inputs.nahual-flake.packages.${system}.nahual)
                  packages.dinov3
                ])
              );
            in
              mkShell {
                packages = [
                  python_with_pkgs
                  python3Packages.venvShellHook
                  pkgs.cudaPackages.cudatoolkit
                  pkgs.cudaPackages.cudnn
                ];
                currentSystem = system;
                venvDir = "./.venv";
                postVenvCreation = ''
                  unset SOURCE_DATE_EPOCH
                '';
                postShellHook = ''
                  unset SOURCE_DATE_EPOCH
                '';
                shellHook = ''
                  runHook venvShellHook
                  # PYTHONSAFEPATH=1 (Python 3.11+) keeps Python from prepending
                  # the script's directory (or cwd for python -c mode) to
                  # sys.path, which would otherwise let the in-tree dinov3/
                  # source dir shadow the nix-built package.
                  export PYTHONSAFEPATH=1
                '';
              };
          };
        }
    );
}
# export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
# export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
# export NVCC_APPEND_FLAGS="-Xcompiler -fno-PIC"
# export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
# export CUDA_NVCC_FLAGS="-O2 -Xcompiler -fno-PIC"
# # Ensure current directory is not in Python path
# export PYTHONDONTWRITEBYTECODE=1
