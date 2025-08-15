{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    # nixpkgs.url = "github:NixOS/nixpkgs/efcb904a6c674d1d3717b06b89b54d65104d4ea7";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
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
      in
      with pkgs;
      rec {
        packages = {
          # xformers = pkgs.python312.pkgs.callPackage ./nix/xformers.nix { };
          dinov2 = pkgs.python312.pkgs.callPackage ./nix/dinov2.nix { };
        };
        devShells = {
          default =
            let
              python_with_pkgs = (
                python312.withPackages (pp: [
                  packages.dinov2
                ])
              );
            in
            mkShell {
              packages = [
                python_with_pkgs
                python312Packages.venvShellHook
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
                # Set PYTHONPATH to only include the Nix packages, excluding current directory
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}
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
