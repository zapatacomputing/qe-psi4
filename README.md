# qe-psi4

[![codecov](https://codecov.io/gh/zapatacomputing/qe-psi4/branch/main/graph/badge.svg?token=RLS534ON7W)](https://codecov.io/gh/zapatacomputing/qe-psi4)

## What is it?

`qe-psi4` is an [Orquestra](https://www.orquestra.io) resource that allows workflows to use the [Psi4 library](http://www.psicode.org).

[Orquestra](https://www.orquestra.io) is a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

## Usage

### Workflow
In order to use `qe-psi4` in your workflow, you need to add it as a `resource` in your Orquestra workflow:

```yaml
imports:
- name: qe-psi4
  type: git
  parameters:
    repository: "git@github.com:zapatacomputing/qe-psi4.git"
    branch: "main"
```

additionally, you will need to add the import in the `imports` argument of your `step`:

```yaml
- name: my-step
  config:
    runtime:
      imports: [qe-psi4]
```

and then set the `language` to `psi4` and the `customImage` to `zapatacomputing/qe-psi4`:

```yaml
- name: my-step
  config:
    runtime:
      imports: [qe-psi4]
      language: psi4
      customImage: "zapatacomputing/qe-psi4"
```

Once that is done you can:
- use any function in the `steps/` directory.
- use tasks which import `qepsi4` in the python code.

## Development and Contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).
