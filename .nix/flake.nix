{
  description = "don't ask nli";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:

  let

    supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    defaultPkgs = forAllSystems (system: import nixpkgs { system=system; });

    tex = pkgs: (pkgs.texlive.combine { 
      inherit (pkgs.texlive) scheme-medium times preprint;
    });

  in

  {
    devShells = forAllSystems (system: {

      default = let pkgs = defaultPkgs.${system}; in pkgs.mkShellNoCC {
        packages = with pkgs; [
          uv
          (tex pkgs) # for writing the paper :)
        ];
       };

    });
  };
}


