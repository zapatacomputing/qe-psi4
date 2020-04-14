# qe-psi4

## What is it?

`qe-psi4` is an [Orquestra](https://www.zapatacomputing.com/orquestra/) resource that allows workflows to use the [Psi4 library](http://www.psicode.org).

[Orquestra](https://www.zapatacomputing.com/orquestra/) is a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

## Usage

### Workflow
In order to use `qe-psi4` in your workflow, you need to add it as a `resource` in your Orquestra workflow:

```yaml
resources:
- name: qe-psi4
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/qe-psi4.git"
    branch: "master"
```

and then import in the `resources` argument of your `task`:

```yaml
- - name: my-task
    template: template-1
    arguments:
      parameters:
      - param_1: 1
      - resources: [qe-psi4]
```

Once that is done you can:
- use any template from the `templates/` directory.
- use tasks which import `qepsi4` in the python code.

## Development and Contribution

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.
