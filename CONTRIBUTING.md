# Contributing
A big welcome and thank you for considering contributing to BEA open source projects! It’s people like you that make it a reality for users in our community.

Reading and following these guidelines will help us make the contribution process easy and effective for everyone involved. It also communicates that you agree to respect the time of the developers managing and developing these open source projects. In return, we will reciprocate that respect by addressing your issue, assessing changes, and helping you finalize your pull requests.

## Code of Conduct

We take our open source community seriously and hold ourselves and other contributors to high standards of communication. By participating and contributing to this project, you agree to uphold our [Code of Conduct](https://github.com/us-bea/.github/blob/main/CODE_OF_CONDUCT.md).

## Getting Started

Contributions are made to this repo via Issues and Pull Requests (PRs). A few general guidelines that cover both:

- To report security vulnerabilities, please see [SECURITY.md](https://github.com/us-bea/beaapi/security/policy).
- Search for existing issues and PRs before creating your own.
- We work hard to make sure issues are handled in a timely manner but, depending on the impact, it could take a while to investigate the root cause. A friendly ping in the comment thread to the submitter or a contributor can help draw attention if your issue is blocking.

## Issues
Issues should be used to report problems with the project, to request new features, or to discuss potential changes before a PR is created.

If you find an issue that addresses the problem you're having, please add your own reproduction information to the existing issue rather than creating a new one. Adding a reaction can also help indicate to our maintainers that a particular problem is affecting more than just the reporter.

## Pull Requests
PRs to our projects are always welcome and can be a quick way to get your fix or improvement slated for the next release. In general, PRs should:

- Only fix/add the functionality in question OR address wide-spread whitespace/style issues, not both.
- Add unit or integration tests for fixed or changed functionality (if a test suite already exists).
- Address a single concern in the least number of changed lines as possible.
- Include documentation in the repo or on our docs site.

For changes that address core functionality or would require breaking changes (e.g. a major release), it's best to open an issue to discuss your proposal first. This is not required but can save time creating and reviewing changes.


## Development
### Building docs
All takes place `docsrc/`. To build the docs you will need `sphinx`. Then:

```
make html
```
which will generate html docs viewable at `docsrc/_build/html/index.html`. We copy `docsrc/_build/html/` to `docs/` 

### Linting
We use `black` and `flake8`.
