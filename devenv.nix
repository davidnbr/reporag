{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  cachix.pull = [ "nixpkgs-python" ];

  packages = [ pkgs.git ];

  languages = {
    python = {
      enable = true;
      venv.enable = true;
    };
  };

  dotenv.enable = true;
}
