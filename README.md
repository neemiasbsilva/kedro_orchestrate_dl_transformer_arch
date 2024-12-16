# Applying Kedro Orchestration Pipelines to Deploy a Deep Learning Transformer Architecture for the Toxic Comment Classification Problem

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Project Overview

This Kedro project, equipped with Kedro-Viz integration, has been developed to analyze and classify textual data from toxic comment datasets. The project's objective is to facilitate machine learning workflows for predicting the toxicity of comments across multiple datasets, leveraging structured pipelines and reproducible data engineering practices. The kedro version `kedro 0.19.10`. For more information take a look at the [Kedro documentation](https://docs.kedro.org).

### Key Features

    - Kedro-Viz Setup: Provides visual representation of the data pipelines to improve workflow comprehension and debugging.
    - Flexible Dataset Management: Handles multiple datasets, including English and multilingual data, designed for toxic comment classification.
    - BERT-Ready Data: Preprocessed datasets for compatibility with transformer-based models like BERT.
    

## Dataset Overview
This project incorporates datasets from the Jigsaw ["Toxic Comment Classification"](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data?select=test.csv). The datasets include labeled comments sourced from platforms like Wikipedia and Civil Comments. 

Description:
    - Comment text: the primary focus, containing the actual text of comments;
    - Toxic Column: Indicates whether the comment is toxic (`1`) or non-toxic (`0`).


## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

### How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

### How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

### Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

#### JupyterLab

To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

#### IPython

And if you want to run an IPython session:

```
kedro ipython
```

#### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

[Further information about using notebooks for experiments within Kedro projects](https://docs.kedro.org/en/develop/notebooks_and_ipython/kedro_and_notebooks.html).

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html).
