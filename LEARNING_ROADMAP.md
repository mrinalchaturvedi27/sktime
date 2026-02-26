# sktime – 7-Day Learning Roadmap

> A structured guide to mastering the sktime codebase for contributors and maintainers.
> Each day has a clear goal, specific files to read, concepts to understand, and hands-on exercises.

---

## Prerequisites

Before starting, make sure you have:

- Python ≥ 3.10 installed
- `git` and a GitHub account
- Familiarity with `pandas`, `numpy`, and basic scikit-learn API (`fit`/`predict`)
- Completed a `pip install sktime` or editable install: `pip install -e ".[dev]"` from the repo root

---

## Overview

| Day | Theme | Goal |
|-----|-------|------|
| 1 | Orientation & Setup | Navigate the repo confidently; run tests locally |
| 2 | Core Abstractions | Understand `BaseObject`, `BaseEstimator`, the tag system |
| 3 | Data Layer | Master mtypes, scitypes, and the conversion framework |
| 4 | Forecasting (flagship) | Read `BaseForecaster` end-to-end; trace a full fit/predict |
| 5 | Other Learning Tasks | Explore classification, transformations, detection, clustering |
| 6 | Pipelines, Composition & Extension | Build, compose, and register a new estimator |
| 7 | Testing, Contributing & Mastery | Write tests, open a PR, and understand the full CI pipeline |

---

## Day 1 – Orientation & Environment Setup

**Goal:** Understand the top-level repository layout, install the project locally, and run the test suite on a small scope.

### Files to read

| File | Why |
|------|-----|
| `README.md` | Project overview, feature table, community links |
| `ARCHITECTURE.md` | Deep architectural reference (this repo); read sections 1–2 today |
| `CONTRIBUTING.md` | Points to the contributing guide on the website |
| `pyproject.toml` | Core dependencies, dev extras, project metadata |
| `setup.cfg` | pytest configuration, test flags (`--matrixdesign`, `--only_changed_modules`) |
| `.pre-commit-config.yaml` | Linting and formatting tools in CI |
| `docs/source/developer_guide/git_workflow.rst` | Forking, branching, and PR workflow |
| `docs/source/developer_guide/coding_standards.rst` | `ruff`, `numpydoc`, naming conventions |

### Concepts to understand

- **Repository layout**: `sktime/` (source), `docs/` (documentation), `examples/` (Jupyter notebooks), `extension_templates/` (contributor templates), `build_tools/` (CI helpers).
- **Core vs soft dependencies**: Core = `numpy`, `pandas`, `scikit-learn`, `scikit-base`; soft = everything else (installed on demand per estimator).
- **Pre-commit hooks**: `ruff` for linting/formatting, `numpydoc` for docstring validation.
- **Test flags**: `--only_changed_modules True` runs only tests for modules touched since the last commit. `--matrixdesign True` subsamples estimators by OS/Python version.

### Hands-on exercises

1. Fork and clone the repo. Create a new branch: `git checkout -b day1-exploration`.
2. Install in editable mode: `pip install -e ".[dev]"`.
   Optional: add `all_extras` (`pip install -e ".[dev,all_extras]"`) to explore all estimators, though this pulls in many optional dependencies and may increase install time.
3. Run a single focused test to confirm your setup:
   ```bash
   python -m pytest sktime/forecasting/naive/tests/ -v --no-header -q
   ```
4. Browse `sktime/` at the top level and write down what each sub-package does (refer to `ARCHITECTURE.md` Section 2).
5. Open `examples/00_sktime_intro.ipynb` in Jupyter and run all cells.

---

## Day 2 – Core Abstractions & Base Classes

**Goal:** Understand the full object hierarchy, the tag/config system, and how `BaseObject`/`BaseEstimator` work.

### Files to read

| File | Why |
|------|-----|
| `sktime/base/_base.py` | `BaseObject` and `BaseEstimator` — the root of everything |
| `sktime/base/_base_panel.py` | `BasePanelMixin` — shared logic for panel estimators |
| `sktime/base/_proba/_base.py` | `BaseDistribution` — probabilistic output objects |
| `sktime/base/_proba/_mixin.py` | `_PredictProbaMixin` — default dispatch for `predict_proba`, `predict_interval`, `predict_quantiles` |
| `sktime/registry/_base_classes.py` | Formal scitype registration via `_BaseScitypeOfObject` |
| `sktime/registry/_tags.py` | Tag definitions (`ESTIMATOR_TAG_REGISTER`) and `_BaseTag` pattern |
| `sktime/registry/__init__.py` | Public registry API: `all_estimators`, `all_tags`, `craft` |
| `sktime/registry/_lookup.py` | `all_estimators()` implementation — how the registry crawls the package |
| `sktime/registry/_craft.py` | `craft()` factory function |
| `ARCHITECTURE.md` sections 3 & 6 | Core abstractions and design patterns |

### Concepts to understand

- **`BaseObject`**: Root class. Provides `get_params`/`set_params`, `clone`, `reset`, and the full tag + config system.
- **`BaseEstimator`**: Adds `is_fitted` property, `check_is_fitted()`, `get_fitted_params()`. Every fittable object (forecaster, classifier, transformer, …) inherits from this.
- **Tag system**: `_tags` class-level dict declaring capabilities. Examples: `"capability:pred_int": True` (can produce prediction intervals), `"y_inner_mtype": "pd.Series"` (expected internal data format). Tags drive conditional behavior in base classes without subclass overrides.
- **Config system**: `_config` dict controls runtime behavior (e.g., `"backend:parallel"`, `"remember_data"`). Set via `set_config(**kwargs)`.
- **`_PredictProbaMixin`**: Default cross-dispatch so that implementing any one of `_predict_interval`, `_predict_quantiles`, `_predict_var`, or `_predict_proba` provides all the others via approximations.
- **Registry**: `all_estimators()` auto-discovers all non-abstract estimators by crawling the `sktime` package. No manual registration is required; just inherit from the right base class and add to `__init__.py`.
- **`craft(name_or_class, **params)`**: Factory that constructs estimators by string name without explicit imports.

### Hands-on exercises

1. Open a Python REPL:
   ```python
   from sktime.registry import all_estimators
   df = all_estimators(as_dataframe=True)
   print(df.shape)          # how many estimators are registered?
   print(df.columns.tolist())
   ```
2. Inspect tags on a concrete estimator:
   ```python
   from sktime.forecasting.naive import NaiveForecaster
   print(NaiveForecaster.get_class_tags())
   print(NaiveForecaster().get_tags())
   ```
3. Experiment with `craft`:
   ```python
   from sktime.registry import craft
   f = craft("NaiveForecaster", strategy="mean", sp=12)
   print(type(f), f.get_params())
   ```
4. Read through the `_PredictProbaMixin._predict_interval` method in `sktime/base/_proba/_mixin.py` and trace how it falls back to `_predict_quantiles` if not implemented.

---

## Day 3 – Data Layer (datatypes, mtypes, conversion)

**Goal:** Master sktime's data type system — the foundation for all estimator input/output.

### Files to read

| File | Why |
|------|-----|
| `sktime/datatypes/__init__.py` | Public API: `check_is_mtype`, `convert_to`, `mtype`, `scitype`, etc. |
| `sktime/datatypes/_registry.py` | `SCITYPE_REGISTER` and `MTYPE_REGISTER` — the master list of all types |
| `sktime/datatypes/_check.py` | `check_is_mtype`, `check_is_scitype`, `mtype()` — validation entry points |
| `sktime/datatypes/_convert.py` | `convert_to()` — the main conversion dispatcher |
| `sktime/datatypes/_series/_mtypes.py` | All mtype definitions for `Series` scitype |
| `sktime/datatypes/_series/_check.py` | Validation logic for Series mtypes |
| `sktime/datatypes/_series/_convert.py` | Conversion functions between Series mtypes |
| `sktime/datatypes/_panel/_base.py` | Panel mtype definitions (`nested_univ`, `numpy3D`, `df-list`, etc.) |
| `sktime/datatypes/_vectorize.py` | `VectorizedDF` — automatic broadcasting of univariate estimators |
| `sktime/datatypes/_utilities.py` | `get_cutoff`, `update_data` |
| `examples/AA_datatypes_and_datasets.ipynb` | Interactive walkthrough of all data formats |
| `ARCHITECTURE.md` section 5 | Data flow and mtype lifecycle |

### Concepts to understand

- **Scitype**: The abstract type of a dataset — `Series`, `Panel`, `Hierarchical`, `Table`, `Alignment`, `Proba`.
- **Mtype (machine type)**: A concrete representation of a scitype. `Series` has `pd.Series`, `pd.DataFrame`, `np.ndarray`. `Panel` has `nested_univ`, `numpy3D`, `pd.DataFrame` with `MultiIndex`, `df-list`.
- **`check_is_mtype(obj, mtype)`**: Returns a boolean or dict of metadata (n_instances, n_columns, n_timepoints, etc.).
- **`convert_to(obj, to_type)`**: Transparent conversion — the workhorse called inside every `fit`/`transform`/`predict`.
- **Inner mtype tags**: `y_inner_mtype` and `X_inner_mtype` on each estimator tell the base class which format to convert inputs to *before* calling `_fit`/`_predict`. Subclass authors write code that assumes this format.
- **`VectorizedDF`**: Wraps a panel or hierarchical dataset and presents it as an iterable of (instance, series) pairs, enabling univariate estimators to be applied element-wise with automatic parallelization.
- **`update_data`**: Appends new observations to existing time series — used by `update()` in forecasters.

### Hands-on exercises

1. Inspect all registered types:
   ```python
   from sktime.datatypes import SCITYPE_REGISTER, MTYPE_REGISTER
   import pandas as pd
   # SCITYPE_REGISTER is a list of (scitype, description) tuples
   print(pd.DataFrame(SCITYPE_REGISTER, columns=["scitype", "description"]))
   # MTYPE_REGISTER is a list of (mtype, scitype, description) tuples
   print(pd.DataFrame(MTYPE_REGISTER, columns=["mtype", "scitype", "description"]))
   ```
2. Practice conversion:
   ```python
   import pandas as pd
   from sktime.datatypes import convert_to, mtype

   series = pd.Series([1, 2, 3, 4, 5], name="y")
   print(mtype(series))               # detected mtype
   arr = convert_to(series, "np.ndarray")
   df = convert_to(series, "pd.DataFrame")
   print(type(arr), type(df))
   ```
3. Create a Panel dataset and check its mtype:
   ```python
   from sktime.utils._testing.panel import make_panel_X
   X = make_panel_X(n_instances=5, n_columns=2, n_timepoints=20)
   from sktime.datatypes import check_is_mtype
   print(check_is_mtype(X, "nested_univ", return_metadata=True))
   ```
4. Trace `convert_to` in `sktime/datatypes/_convert.py` — find where it dispatches to the actual converter function.

---

## Day 4 – Forecasting (the Flagship Module)

**Goal:** Read `BaseForecaster` end-to-end. Trace a complete fit/predict/predict_proba pipeline. Understand `ForecastingHorizon`.

### Files to read

| File | Why |
|------|-----|
| `sktime/forecasting/base/_base.py` | **`BaseForecaster`** — the central class; read the full `fit`, `predict`, `predict_interval`, `predict_proba` methods |
| `sktime/forecasting/base/_fh.py` | `ForecastingHorizon` — relative/absolute horizon, cutoff, time delta |
| `sktime/forecasting/base/_sktime.py` | `_BaseWindowForecaster` — window-based forecaster mixin |
| `sktime/forecasting/naive/_naive.py` | `NaiveForecaster` — simplest concrete forecaster; great for tracing |
| `sktime/forecasting/compose/_pipeline.py` | `TransformedTargetForecaster`, `ForecastingPipeline` |
| `sktime/forecasting/model_selection/_base.py` | `BaseGridSearch` |
| `sktime/forecasting/model_selection/_gridsearch.py` | `ForecastingGridSearchCV` |
| `sktime/forecasting/base/adapters/_statsmodels.py` | Adapter pattern for wrapping statsmodels |
| `sktime/forecasting/base/adapters/_pytorch.py` | `BaseDeepNetworkPyTorch` |
| `sktime/base/_proba/_base.py` | `BaseDistribution` — return type of `predict_proba` |
| `sktime/base/_proba/_normal.py` | `Normal` — concrete distribution example |
| `examples/01_forecasting.ipynb` | Forecasting tutorial |
| `examples/01b_forecasting_proba.ipynb` | Probabilistic forecasting tutorial |
| `examples/01c_forecasting_hierarchical_global.ipynb` | Hierarchical & global forecasting |
| `extension_templates/forecasting.py` | Complete extension template with all docstrings |
| `ARCHITECTURE.md` section 4 | Model execution flow |

### Concepts to understand

- **`ForecastingHorizon` (fh)**: Represents the time steps to forecast. Can be relative (integers like `[1,2,3]`) or absolute (datetime index). Stored on `self._fh` after `fit`.
- **Cutoff**: `self._cutoff` — the last observed time point after fitting; fh is resolved relative to it.
- **Public vs private methods**: `fit` → `_fit`, `predict` → `_predict`. The public methods handle validation, type conversion, and state management. The private methods contain algorithm logic.
- **Inner type conversion in fit**: `y` and `X` are converted to `y_inner_mtype` / `X_inner_mtype` before `_fit` is called. Subclasses receive already-converted data.
- **Prediction intervals**: `predict_interval(fh, coverage)` → `_predict_interval` → internally may call `_predict_quantiles`.
- **Probabilistic forecasting**: `predict_proba(fh)` → `_predict_proba` → returns a `BaseDistribution` instance (e.g., `Normal(mu=..., sigma=...)`).
- **Update**: `update(y_new)` ingests new data without full refit (if `update_params=False`). Base class merges new data into `self._y`.
- **Adapters**: `sktime/forecasting/base/adapters/` wraps external libraries (statsmodels, pmdarima, Prophet, PyTorch) behind `BaseForecaster`. The adapter pattern means users get the full sktime interface.
- **`_BaseGlobalForecaster`**: Experimental extension to support panel/hierarchical fitting in a single model (e.g., neural network trained on all instances).

### Hands-on exercises

1. Trace `NaiveForecaster.fit` step by step. The simplest approach is to add temporary `print` statements inside `BaseForecaster.fit` and `NaiveForecaster._fit` to watch the flow:
   ```python
   from sktime.forecasting.naive import NaiveForecaster
   import pandas as pd
   y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
   f = NaiveForecaster(strategy="last")
   f.fit(y)
   print("cutoff:", f.cutoff)
   print("is_fitted:", f.is_fitted)
   ```
   Advanced alternative: use `pdb.set_trace()` inside `BaseForecaster.fit` or an IDE debugger with a breakpoint.
2. Predict intervals and a full distribution:
   ```python
   from sktime.forecasting.naive import NaiveForecaster
   import pandas as pd
   y = pd.Series(range(30))
   f = NaiveForecaster(strategy="drift")
   f.fit(y)
   print(f.predict([1, 2, 3]))
   print(f.predict_interval([1, 2, 3], coverage=0.9))
   print(f.predict_proba([1, 2, 3]))
   ```
3. Build a pipeline:
   ```python
   from sktime.forecasting.compose import TransformedTargetForecaster
   from sktime.transformations.series.difference import Differencer
   from sktime.forecasting.naive import NaiveForecaster
   pipe = TransformedTargetForecaster([
       ("diff", Differencer()),
       ("naive", NaiveForecaster(strategy="last")),
   ])
   pipe.fit(y)
   print(pipe.predict([1, 2, 3]))
   ```
4. Run model selection:
   ```python
   from sktime.forecasting.model_selection import ForecastingGridSearchCV, SlidingWindowSplitter
   from sktime.forecasting.naive import NaiveForecaster
   cv = SlidingWindowSplitter(window_length=10, fh=[1, 2, 3])
   gs = ForecastingGridSearchCV(NaiveForecaster(), cv=cv, param_grid={"strategy": ["last", "mean", "drift"]})
   gs.fit(y)
   print(gs.best_params_)
   ```

---

## Day 5 – Other Learning Tasks

**Goal:** Understand the API, base classes, and key patterns for transformations, classification, regression, clustering, and detection.

### 5a – Transformations

#### Files to read

| File | Why |
|------|-----|
| `sktime/transformations/base.py` | `BaseTransformer` — `fit`, `transform`, `fit_transform`, `inverse_transform`; mtype dispatch |
| `sktime/transformations/series/difference.py` | Simple concrete transformer; great for tracing |
| `sktime/transformations/series/detrend/__init__.py` | Detrending transformer |
| `sktime/transformations/compose/_pipeline.py` | `TransformationPipeline` |
| `sktime/transformations/compose/_featureunion.py` | `FeatureUnion` for parallel transformers |
| `extension_templates/transformer.py` | Transformer extension template |
| `examples/03_transformers.ipynb` | Transformations tutorial |

#### Concepts to understand

- `BaseTransformer` handles three scitype combinations: Series→Series, Series→Primitives (tabularisation), Panel→Panel, Panel→Table.
- `transform` returns a transformed dataset of the same or different scitype.
- `inverse_transform` is optional (tag `"capability:inverse_transform": True`).
- Transformers can be used standalone or inside pipelines (both forecasting and classification pipelines accept them).

### 5b – Time Series Classification & Regression

#### Files to read

| File | Why |
|------|-----|
| `sktime/classification/base.py` | `BaseClassifier` — `fit`, `predict`, `predict_proba` for panel data |
| `sktime/classification/compose/_pipeline.py` | `ClassifierPipeline` |
| `sktime/classification/distance_based/_time_series_neighbors.py` | KNN classifier — a clean, readable concrete example |
| `sktime/regression/base.py` | `BaseRegressor` — same pattern as `BaseClassifier` |
| `sktime/classification/deep_learning/base/_base_tf.py` | `BaseDeepClassifier` (Keras) |
| `extension_templates/classification.py` | Classifier extension template |
| `examples/02_classification.ipynb` | Classification tutorial |

#### Concepts to understand

- Both `BaseClassifier` and `BaseRegressor` inherit `BasePanelMixin` which handles panel data input.
- `predict_proba` in classifiers returns class probability arrays (not distributions), unlike forecasters.
- Deep learning classifiers (`BaseDeepClassifier`) build a Keras model via `build_model(input_shape, n_classes)` and wrap `fit`/`predict` around the Keras training loop.

### 5c – Clustering

#### Files to read

| File | Why |
|------|-----|
| `sktime/clustering/base.py` | `BaseClusterer` — `fit`, `predict`, `fit_predict`, `predict_proba` |
| `sktime/clustering/k_means/_k_means.py` | K-Means clusterer — concrete example |
| `sktime/clustering/partitioning/_lloyds.py` | `BaseTimeSeriesLloyds` — Lloyd's algorithm mixin |

#### Concepts to understand

- `BaseClusterer` operates on panel data (multiple time series).
- `predict` assigns cluster labels; `predict_proba` gives soft assignments.
- Distances (`sktime/distances/`) are used internally by many clusterers.

### 5d – Anomaly / Change-point Detection

#### Files to read

| File | Why |
|------|-----|
| `sktime/detection/base/_base.py` | `BaseDetector` — `fit`, `predict`, `transform`, `fit_predict` |
| `sktime/detection/clasp.py` | CLASP change-point detector — active example |
| `sktime/detection/naive/_threshold.py` | Naive threshold detector baseline |
| `extension_templates/detection.py` | Detection extension template |
| `examples/07_detection_anomaly_changepoints.ipynb` | Detection tutorial |

#### Concepts to understand

- `BaseDetector` has both `predict` (returns labeled index) and `transform` (returns binary/score Series).
- Detection is the newest major task in sktime; the API is still evolving.
- Anomaly detection and change-point detection share the same base class but differ in output semantics (tags differentiate them).

### 5e – Distances & Kernels

#### Files to read

| File | Why |
|------|-----|
| `sktime/dists_kernels/base/_base.py` | `BasePairwiseTransformer`, `BasePairwiseTransformerPanel` |
| `sktime/distances/` | Pure-function distance implementations (DTW, LCSS, ERP, etc.) |
| `examples/06_distances_kernels_alignment.ipynb` | Tutorial |

### Hands-on exercises

1. Apply a transformer and inspect output type:
   ```python
   from sktime.transformations.series.difference import Differencer
   import pandas as pd
   s = pd.Series([1.0, 3.0, 6.0, 10.0, 15.0])
   t = Differencer()
   print(t.fit_transform(s))
   ```
2. Classify time series:
   ```python
   from sktime.datasets import load_arrow_head
   from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
   X_train, y_train = load_arrow_head(split="train")
   clf = KNeighborsTimeSeriesClassifier(n_neighbors=1)
   clf.fit(X_train, y_train)
   X_test, y_test = load_arrow_head(split="test")
   print(clf.score(X_test, y_test))
   ```

---

## Day 6 – Pipelines, Composition & Extension Mechanisms

**Goal:** Understand all composition patterns and implement your first custom estimator end-to-end.

### Files to read

| File | Why |
|------|-----|
| `sktime/pipeline/pipeline.py` | Generic pipeline primitives |
| `sktime/forecasting/compose/_pipeline.py` | `TransformedTargetForecaster`, `ForecastingPipeline`, `ForecastX` |
| `sktime/forecasting/compose/_multiplexer.py` | `MultiplexForecaster` — strategy pattern for model selection |
| `sktime/forecasting/compose/_ensemble.py` | Ensemble of forecasters |
| `sktime/transformations/compose/_featureunion.py` | Parallel feature construction |
| `sktime/base/_meta.py` | `_HeterogenousMetaEstimator` — base for all composite estimators |
| `sktime/utils/dependencies/_dependencies.py` | `_check_soft_dependencies`, `_check_estimator_deps` |
| `extension_templates/forecasting.py` | **Read this in full** — the definitive guide to implementing a forecaster |
| `extension_templates/transformer.py` | Transformer template |
| `extension_templates/classification.py` | Classifier template |
| `docs/source/developer_guide/add_estimators.rst` | Step-by-step contributor guide |
| `docs/source/developer_guide/dependencies.rst` | How to declare and isolate soft dependencies |
| `ARCHITECTURE.md` section 12 | Extension mechanisms (exact steps) |

### Concepts to understand

- **`_HeterogenousMetaEstimator`**: Base for pipelines and ensembles. Provides `get_params`/`set_params` that drill down into component estimators.
- **Pipeline construction**: `TransformedTargetForecaster([("t1", Transformer()), ("f", Forecaster())])` — the `__mul__` and `__or__` operators are overloaded so you can write `Transformer() * Forecaster()`.
- **`MultiplexForecaster`**: Wraps a list of forecasters and switches between them based on a `selected_forecaster` parameter — key for model selection.
- **Soft dependency pattern**: Import inside the method (`_fit`, `_predict`, `__init__`), not at module level. Declare via tag: `"python_dependencies": ["statsmodels>=0.12"]`.
- **`get_test_params`**: Class method returning a list of parameter dicts. Required for the test framework to instantiate and validate your estimator.

### Hands-on exercises

1. **Implement a custom forecaster** using the template:
   ```bash
   cp extension_templates/forecasting.py /tmp/my_forecaster.py
   ```
   Edit `/tmp/my_forecaster.py`:
   - Implement `_fit` to store the mean of `y`.
   - Implement `_predict` to return that mean for all forecast steps.
   - Implement `get_test_params` returning `[{}]`.

2. **Run `check_estimator`**:
   ```python
   import sys
   sys.path.insert(0, "/tmp")
   from my_forecaster import MyForecaster
   from sktime.utils.estimator_checks import check_estimator
   check_estimator(MyForecaster, raise_exceptions=True)
   ```

3. **Build a complex pipeline**:
   ```python
   from sktime.forecasting.compose import TransformedTargetForecaster, ForecastingPipeline
   from sktime.transformations.series.difference import Differencer
   from sktime.transformations.series.detrend import Detrender
   from sktime.forecasting.naive import NaiveForecaster
   from sktime.transformations.series.impute import Imputer
   pipe = TransformedTargetForecaster([
       ("impute", Imputer(method="mean")),
       ("detrend", Detrender()),
       ("diff", Differencer()),
       ("forecast", NaiveForecaster(strategy="last")),
   ])
   import pandas as pd
   y = pd.Series([1.0, 2.0, None, 4.0, 5.0, 6.0])
   pipe.fit(y)
   print(pipe.predict([1, 2, 3]))
   ```

4. **Operator syntax**:
   ```python
   from sktime.transformations.series.difference import Differencer
   from sktime.forecasting.naive import NaiveForecaster
   pipe = Differencer() * NaiveForecaster(strategy="last")
   print(type(pipe))
   ```

---

## Day 7 – Testing, Contributing & Mastery

**Goal:** Understand the three-layer test architecture, run and extend the test suite, and complete a full contribution cycle.

### Files to read

| File | Why |
|------|-----|
| `sktime/tests/test_all_estimators.py` | Layer 1: interface compliance tests for ALL estimators |
| `sktime/forecasting/tests/test_all_forecasters.py` | Layer 2: forecaster-specific interface tests |
| `sktime/tests/_config.py` | `EXCLUDE_ESTIMATORS`, `EXCLUDED_TESTS`, test configuration |
| `sktime/tests/test_switch.py` | `run_test_for_class` — conditional test execution |
| `sktime/utils/_testing/estimator_checks.py` | `check_estimator` utility |
| `sktime/utils/_testing/scenarios_forecasting.py` | Test scenario definitions for forecasters |
| `sktime/utils/_testing/scenarios_getter.py` | Scenario retrieval by scitype |
| `sktime/utils/estimator_checks.py` | Public `check_estimator` function |
| `docs/source/developer_guide/testing_framework.rst` | Testing framework overview (architecture, how to add tests) |
| `docs/source/developer_guide/add_estimators.rst` | How to contribute an estimator (end-to-end) |
| `docs/source/developer_guide/continuous_integration.rst` | CI pipeline and workflow structure |
| `docs/source/developer_guide/deprecation.rst` | How to deprecate and migrate API |
| `ARCHITECTURE.md` section 8 | Testing framework summary |
| `ARCHITECTURE.md` section 13 | Contribution entry points |

### Concepts to understand

**Test layer architecture:**
- **Layer 1** (`test_all_estimators.py`): Tests that apply to **every object** in sktime. Checks parameter handling, tag validity, `clone`, `reset`, serialization, HTML repr.
- **Layer 2** (`test_all_[task]s.py`): Tests that apply to **every estimator of a specific scitype** (e.g., all forecasters). Checks `fit`/`predict` contracts, output shapes, index consistency.
- **Layer 3** (per-module `tests/` folders): Algorithm-specific unit tests (edge cases, numerical correctness, specific inputs).

**`get_test_params` contract:**
- Must return a list of dicts, each dict being a valid set of constructor parameters.
- The test framework instantiates the estimator with each dict and runs all applicable Layer 1 and Layer 2 tests.
- Providing diverse parameter sets (e.g., different strategies, different window sizes) improves coverage.

**`run_test_for_class`:**
- Gates test execution based on: estimator tags (`tests:skip_all`, `tests:skip_by_name`), whether the estimator's soft dependencies are available, and CI configuration.

**CI workflow:**
- On PR, `--only_changed_modules True` is set — only tests for modules touched by the PR are run.
- Full test matrix runs on merge to `main`.
- Subsampling by OS/Python version ensures each estimator is tested on every platform over time.

**Deprecation process:**
- Use `FutureWarning` with a message specifying the removal version.
- Keep the old API working for at least one minor version before removal.
- Document in `CHANGELOG.md`.

### Hands-on exercises

1. **Run Layer 1 tests on your custom estimator:**
   ```bash
   python -m pytest sktime/tests/test_all_estimators.py -k "MyForecaster" -v
   ```

2. **Run Layer 2 tests on a real forecaster:**
   ```bash
   python -m pytest sktime/forecasting/tests/test_all_forecasters.py -k "NaiveForecaster" -v --no-header -q
   ```

3. **Add a Layer 3 unit test.** Create `sktime/forecasting/naive/tests/test_naive_custom.py`:
   ```python
   import pandas as pd
   import pytest
   from sktime.forecasting.naive import NaiveForecaster

   def test_naive_mean_prediction():
       y = pd.Series([2.0, 4.0, 6.0])
       f = NaiveForecaster(strategy="mean")
       f.fit(y)
       pred = f.predict([1])
       assert abs(pred.iloc[0] - 4.0) < 1e-6, f"Expected 4.0, got {pred.iloc[0]}"
   ```
   Run it:
   ```bash
   python -m pytest sktime/forecasting/naive/tests/test_naive_custom.py -v
   ```

4. **Full contribution cycle (simulate):**
   - Pick a `good first issue` or `help wanted` issue from the GitHub tracker.
   - Create a branch: `git checkout -b my-contribution`.
   - Implement the change.
   - Run `pre-commit run --all-files` to check formatting.
   - Run `check_estimator(MyEstimator)` if adding a new estimator.
   - Run targeted tests: `python -m pytest sktime/[module]/tests/ -v`.
   - Commit and open a draft PR.

5. **Read `CHANGELOG.md`** — focus on the most recent 2–3 releases to understand what is actively being developed and what the contribution bar looks like.

---

## Key Concepts Reference Card

| Concept | Location | One-liner |
|---------|----------|-----------|
| `BaseObject` | `sktime/base/_base.py` | Root of all sktime objects; tags, configs, params |
| `BaseEstimator` | `sktime/base/_base.py` | Adds `is_fitted`, `get_fitted_params` |
| `BaseForecaster` | `sktime/forecasting/base/_base.py` | Central forecasting interface |
| `ForecastingHorizon` | `sktime/forecasting/base/_fh.py` | Encapsulates time steps to forecast |
| `BaseTransformer` | `sktime/transformations/base.py` | fit/transform interface for time series |
| `BaseClassifier` | `sktime/classification/base.py` | Panel time series classification |
| `BaseClusterer` | `sktime/clustering/base.py` | Panel time series clustering |
| `BaseDetector` | `sktime/detection/base/_base.py` | Anomaly/change-point detection |
| `BaseDistribution` | `sktime/base/_proba/_base.py` | Probabilistic output object |
| `_PredictProbaMixin` | `sktime/base/_proba/_mixin.py` | Default dispatch for probabilistic methods |
| `VectorizedDF` | `sktime/datatypes/_vectorize.py` | Broadcasts univariate estimator over panel |
| `all_estimators` | `sktime/registry/_lookup.py` | Programmatic discovery of all estimators |
| `check_estimator` | `sktime/utils/estimator_checks.py` | Full interface compliance test for one estimator |
| `convert_to` | `sktime/datatypes/_convert.py` | Convert data between mtypes |
| Mtype tags | `sktime/datatypes/_registry.py` | Catalogue of all data types |
| Tag system | `sktime/registry/_tags.py` | Capability declarations on estimators |
| Extension templates | `extension_templates/` | Fill-in templates for new estimators |

---

## Recommended Reading Order Beyond Day 7

Once you have completed the 7-day roadmap, deepen your expertise in these areas:

1. **Hierarchical forecasting**: `sktime/transformations/hierarchical/`, `examples/01c_forecasting_hierarchical_global.ipynb`
2. **Probabilistic forecasting**: `sktime/proba/` (distribution library), `examples/01b_forecasting_proba.ipynb`
3. **Deep learning integration**: `sktime/networks/`, `sktime/classification/deep_learning/`, `sktime/forecasting/base/adapters/_pytorch.py`
4. **Distances and kernels**: `sktime/distances/`, `examples/06_distances_kernels_alignment.ipynb`
5. **Benchmarking**: `sktime/benchmarking/`, `examples/04_benchmarking_forecasters.ipynb`
6. **Global forecasting**: `sktime/forecasting/base/_base.py` class `_BaseGlobalForecaster`
7. **Architecture deep-dive**: Re-read `ARCHITECTURE.md` in full — by Day 7 you will understand every section at implementation level.

---

## Community & Staying Current

| Resource | Link |
|----------|------|
| Discord (main community hub) | https://discord.com/invite/54ACzaFsn7 |
| GitHub Discussions | https://github.com/sktime/sktime/discussions |
| Weekly dev meet-up | Fridays 13:00 UTC, `dev/meet-ups` channel on Discord |
| Issue tracker (good first issues) | https://github.com/sktime/sktime/issues?q=label%3A%22good+first+issue%22 |
| Release notes | `CHANGELOG.md` in this repo |
| Roadmap | `docs/source/roadmap.rst` |
