{
  description = "mixup using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python313;

          # The CUDA wheels have inter-NVIDIA and RDMA/MPI/fabric dependencies
          # (libcuda, libnvJitLink, libcusparse, libcublas, libmlx5, libmpi, ...)
          # that aren't in each wheel's own build closure. They all resolve at
          # runtime (torch loads the sibling NVIDIA libs itself, and we never use
          # the RDMA/MPI transports), so tell autoPatchelf to ignore missing deps.
          cudaWheelFix = final: prev: {
            torch = prev.torch.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-cufile = prev.nvidia-cufile.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-nvshmem-cu13 = prev.nvidia-nvshmem-cu13.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-cusparse = prev.nvidia-cusparse.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-cusolver = prev.nvidia-cusolver.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-cudnn-cu13 = prev.nvidia-cudnn-cu13.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-cusparselt-cu13 = prev.nvidia-cusparselt-cu13.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
            nvidia-nccl-cu13 = prev.nvidia-nccl-cu13.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps =
                (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "*" ];
            });
          };
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              cudaWheelFix
            ]
          )
      );
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          # mixup-node (torch + CUDA) runs inside Docker containers, not in the host
          # dev shell, so exclude it here. This keeps `nix develop` off the CUDA
          # wheels, which don't build on NixOS (nvidia-cufile -> rdma-core).
          hostDeps = builtins.removeAttrs workspace.deps.all [ "mixup-node" ];
          virtualenv = pythonSet.mkVirtualEnv "mixup-dev-env" hostDeps;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );
    };
}
