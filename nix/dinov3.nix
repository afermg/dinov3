{
  lib,
  # build deps
  buildPythonPackage,
  fetchFromGitHub,
  # Py build
  setuptools,
  # Deps
  torch,
  torchvision,
  omegaconf,
  torchmetrics,
  fvcore,
  iopath,
  # xformers,
  transformers,
  submitit,
  # Extras
  mmcv,
  # Server,
  datasets,
  scikit-learn,
  regex,
  ftfy,
  matplotlib,
}:
buildPythonPackage {
  pname = "dinov3";
  version = "0.3.2";

  src = ./..; # For local testing, add flag --impure when running
  # src = fetchFromGitHub {
  #   owner = "afermg";
  #   repo = "baby";
  #   rev = "39eec0d4c3b8fad9b0a8683cbedf9b4558e07222";
  #   sha256 = "sha256-ptLXindgixDa4AV3x+sQ9I4W0PScIQMkyMNMo0WFa0M=";
  # };

  pyproject = true;
  buildInputs = [
    # setuptools-scm
    setuptools
  ];
  propagatedBuildInputs = [
    torch
    torchvision
    omegaconf
    torchmetrics
    fvcore
    iopath
    # xformers
    submitit
    # extras
    # mmcv
    # Server
    datasets
    scikit-learn
    regex
    ftfy
    transformers
    matplotlib
  ];

  pythonImportsCheck = [
  ];

  meta = {
    description = "dinov2";
    homepage = "https://github.com/afermg/dinov3";
    license = lib.licenses.asl20;
  };
}
