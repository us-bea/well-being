# BEA Wellbeing
## Project Description
This repository provides the code to generate the charts and tables for the BEA's [Prototype Measures of Economic Well-Being and Growth](https://apps.bea.gov/well-being/) as well as a Jupyter notebook to view/modify them. 

## Setup
1. This notebook requires Python >=3.9 and the packages listed in [requirements.txt](requirements.txt) (versions listed are known to work, though other versions may work as well). The `beaapi` package requires manual installation, so download the latest `.whl` file from the [releases page](https://github.com/us-bea/beaapi/releases) and install using `pip`.
2. Get [FRED](https://fred.stlouisfed.org/docs/api/api_key.html), [BLS](https://data.bls.gov/registrationEngine/), and [BEA](https://apps.bea.gov/API/signup/) api keys. Copy `apiKeys.default.env` to `apiKeys.env` and insert your keys after the `=` for each service.


## Running
Open the Jupyter notebook [wellbeing.ipynb](wellbeing.ipynb), edit if you'd like, and run!

For help, see our [documention](https://us-bea.github.io/well-being/).

## Quick Links
* [CODE_OF_CONDUCT.md](https://github.com/us-bea/.github/blob/main/CODE_OF_CONDUCT.md)
* [SECURITY.md](https://github.com/us-bea/beaapi/security/policy)
* [LICENSE](LICENSE)
* [CONTRIBUTING.md](CONTRIBUTING.md)
* [SUPPORT.md](SUPPORT.md)
<!-- * [CHANGELOG.md](CHANGELOG.md) -->