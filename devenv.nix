{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  cachix.pull = [ "nixpkgs-python" ];

  packages = [ pkgs.git pkgs.zlib pkgs.stdenv.cc.cc.lib ];

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.zlib pkgs.stdenv.cc.cc.lib ];

  languages = {
    python = {
      enable = true;
      venv.enable = true;
    };
  };

  dotenv.enable = true;
}
