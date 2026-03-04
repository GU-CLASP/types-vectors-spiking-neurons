nixpkgs_source: self: pkgs:
         with pkgs; {
           linuxPackages = pkgs.linuxPackages.extend (self: super: {
              nvidia_x11 = callPackage (import (nixpkgs_source + "/pkgs/os-specific/linux/nvidia-x11/generic.nix") {
                version = "555.42.02";
                sha256_64bit = "sha256-k7cI3ZDlKp4mT46jMkLaIrc2YUx1lh1wj/J4SVSHWyk=";
                settingsSha256 = "1677g7rcjbcs5fja1s4p0syhhz46g9x2qqzyn3wwwrjsj7rwbz78";
                persistencedSha256 = "01kvd3zp056i4n8vazj7gx1xw0h4yjdlpazmspnsmwg23ijb83x4";
              }) {
                libsOnly = true;
              };
            });
          }
