{
  description = "Vector Symolic Architectures / Type Theory with Records";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:

  let

    supportedSystems = [ 
      "x86_64-linux" 
      "x86_64-darwin" 
      "aarch64-linux" 
      "aarch64-darwin" 
    ];

    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

    pythonOverlay = (final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [ 
        (pyFinal: pyPrev: {
          torchhd = final.python3.pkgs.callPackage ./torchhd.nix { inherit (pyFinal.pkgs); };
        })
      ];
    });

    localPkgs = forAllSystems (system: import nixpkgs { 
      system=system; 
      overlays=[ pythonOverlay ]; 
    });

    serverPkgs = forAllSystems (system: import nixpkgs { 
      system=system; 
      overlays=[ 
        pythonOverlay 
        # fix version of nvidia drivers
        ((import ./nvidia-555.42.02.nix) nixpkgs)  
      ]; 
      config={
        cudaSupport = true;
        cudaVersion = "12.5";
        allowUnfree = true;
        nvidia.acceptLicense = true;
      };
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

    texPackages = (ps: with ps; [
      latexmk
      cm-super
      biblatex
      csquotes 
      numprint 
      mathtools 
      expex 
      minimalist 
      listings 
      tree-dvips 
      gb4e 
      was 
      pbox 
      relsize 
      algpseudocodex 
      algorithmicx 
      inconsolata 
      upquote 
      adjustbox 
      covington 
      varwidth
    ]);

    packages = (pkgs: with pkgs; [
      (python3.withPackages pythonPackages)
      (texliveSmall.withPackages texPackages)
      biber
      typst
      quarto
      bash
      entr
      codex
    ]);

  in
  {

    devShells = forAllSystems (system: {

      default = let pkgs = localPkgs.${system}; in pkgs.mkShellNoCC {
        packages =  (packages pkgs);
        shellHook = ''
          export QUARTO_PYTHON=$(which python3)
        '';
      };

      server = let pkgs = serverPkgs.${system}; in pkgs.mkShellNoCC {
        packages =  (packages pkgs);
        shellHook = ''
          export QUARTO_PYTHON=$(which python3)
          export CUDA_PATH=${pkgs.cudatoolkit}
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudatoolkit}/lib"
          export HF_HOME=~/.cache/huggingface
        '';
      };

    });

  };

}
