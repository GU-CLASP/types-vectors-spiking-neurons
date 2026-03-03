{
  description = "Vector Symolic Architectures / Type Theory with Records";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:

  let

    supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    pkgs = forAllSystems (system: import nixpkgs { system=system; overlays=overlays; });

    overlays = [
      (final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [ 
          (pyFinal: pyPrev: {
            torchhd = final.python3.pkgs.callPackage ./torchhd.nix { inherit (pyFinal.pkgs); };
          })
        ];
      })

    ];

    tex = pkgs: (pkgs.texlive.combine { 
      inherit (pkgs.texlive) scheme-medium csquotes numprint mathtools expex ;
    });

    pythonPackages = (ps: with ps; [
      jupyter
      ipython
      pandas
      torch
      torchhd
      torchmetrics
      torchvision
      pillow
      tqdm
      matplotlib
      numpy
      scikit-learn
    ]);

    packages = (pkgs: with pkgs; [
      (python3.withPackages pythonPackages)
      (tex pkgs)
      typst
      quarto
      bash
    ]);

  in
  {

    devShells = forAllSystems (system: {
      default = pkgs.${system}.mkShellNoCC {
        packages =  (packages pkgs.${system});
        shellHook = ''
          export QUARTO_PYTHON=$(which python3)
        '';
      };
    });

  };

}
