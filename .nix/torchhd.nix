{ lib
, setuptools
, buildPythonPackage
, fetchFromGitHub
, torch
, pandas
, tqdm
, openpyxl
, scipy
}:

buildPythonPackage rec {
  pname = "torchhd";
  version = "v5.8.4";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "hyperdimensional-computing";
    repo = "torchhd";
    rev = "refs/tags/${version}";
    hash = "sha256-7bUgLf6ZKP9HGAOjN6LeVQ9Dw0v7N4RbAW1rRYgBvBY=";
  };

  nativeBuildInputs = [
  ];

  build-system = [
    setuptools
  ];

  dependencies = [
    torch
    pandas
    tqdm
    openpyxl
    scipy
  ];

  nativeCheckInputs = [
  ];

  pythonImportsCheck = [ "torchhd" ];

}

