{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs.python311Packages; [
    numpy
    numba
  ];

  languages.python = {
    enable = true;
    package = pkgs.python311;
    directory = "./src";
    venv.enable = true;
    venv.requirements = ''
        tensorflow
        tensorflow-datasets
    '';
    uv.enable = true;
  };
}
