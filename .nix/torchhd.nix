{ lib
, setuptools
, buildPythonPackage
, fetchFromGitHub
, numpy
, torch
, pandas
, tqdm
, openpyxl
, scipy
, requests
}:

buildPythonPackage rec {
  pname = "torch-hd";
  version = "5.8.4";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "hyperdimensional-computing";
    repo = "torchhd";
    rev = "refs/tags/v${version}";
    hash = "sha256-7bUgLf6ZKP9HGAOjN6LeVQ9Dw0v7N4RbAW1rRYgBvBY=";
  };

  nativeBuildInputs = [
  ];

  build-system = [
    setuptools
  ];

  dependencies = [
    numpy
    torch
    pandas
    tqdm
    openpyxl
    scipy
    requests
  ];

  nativeCheckInputs = [
  ];

  pythonImportsCheck = [ "torchhd" ];

}

