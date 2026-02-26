# sktime – Deep Architectural Analysis

> A comprehensive guide for core contributors and future maintainers.

---

## Table of Contents

1. [High-Level Purpose](#1-high-level-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Abstractions](#3-core-abstractions)
4. [Model Execution Flow](#4-model-execution-flow)
5. [Data Flow](#5-data-flow)
6. [Design Patterns](#6-design-patterns)
7. [Dependency Structure](#7-dependency-structure)
8. [Testing Framework](#8-testing-framework)
9. [Active Development Zones](#9-active-development-zones)
10. [Complexity & Technical Debt](#10-complexity--technical-debt)
11. [Performance Analysis](#11-performance-analysis)
12. [Extension Mechanisms](#12-extension-mechanisms)
13. [Contribution Entry Points](#13-contribution-entry-points)
14. [Roadmap Signals](#14-roadmap-signals)
15. [Strategic Summary](#15-strategic-summary)

---

## 1. High-Level Purpose

### What problem does this repository solve?

**sktime** is a unified Python library for machine learning with time series. It solves the fundamental problem that the time series machine learning ecosystem is fragmented: different tools use incompatible data formats, APIs, and conventions, making it painful to combine algorithms, benchmark them against each other, or build production pipelines. sktime provides:

- A **common interface** (API contract) for all time series learning tasks.
- **Data conversion** infrastructure so algorithms that expect different internal formats can interoperate transparently.
- **Composable pipelines** that chain transformers, forecasters, and classifiers.
- A **registry** of hundreds of estimators, enabling easy discovery and benchmarking.

### What is its role in the larger ecosystem?

sktime occupies the same role for time series that scikit-learn occupies for tabular data. It is the de-facto standard Python library for time series ML and is explicitly modelled on the scikit-learn API. It integrates with:

- **scikit-learn** – inherits from and wraps sklearn base classes; sklearn estimators can be used inside sktime pipelines.
- **statsmodels / pmdarima / Prophet / statsforecast** – wrapped as sktime forecasters.
- **TensorFlow / Keras** and **PyTorch** – wrapped via dedicated deep learning base adapters.
- **skbase** – provides the low-level `BaseObject` and `BaseEstimator` primitives that sktime extends.
- **darts, neuralforecast, GluonTS, TBATS** – wrapped or adapted as sktime estimators.

### What type of users is it designed for?

| User type | Usage |
|-----------|-------|
| Applied data scientists | High-level API: `fit`, `predict`, pipelines |
| ML researchers | Implement new algorithms via extension templates |
| Software engineers / MLOps | Serialisation, deployment, reproducibility |
| Package contributors / maintainers | Tag system, registry, testing framework |

### What design philosophy does it follow?

- **scikit-learn compatible** – same `fit/predict` convention, `get_params/set_params`, `clone`.
- **Task-oriented** – each learning task (forecasting, classification, etc.) has its own base class and interface contract.
- **Type-safe via tags** – estimators declare their capabilities and input/output types through a tag system rather than hard-coded checks.
- **Internal type conversion** – user data is automatically converted to the estimator's preferred internal representation; users need not worry about format.
- **Minimal mandatory surface** – only the truly task-specific methods (`_fit`, `_predict`, …) must be implemented; everything else is provided by the base class.

---

## 2. Architecture Overview

### Major modules / packages

| Module | Responsibility |
|--------|---------------|
| `sktime/base` | Root base classes (`BaseObject`, `BaseEstimator`, `BaseDistribution`) and shared mixins |
| `sktime/registry` | Global estimator registry, scitype lookup, tag definitions, `all_estimators()` |
| `sktime/datatypes` | Data-type definitions (mtypes/scitypes), converters, validators, examples |
| `sktime/forecasting` | Forecasting estimators, base class, horizon (`ForecastingHorizon`), model selection |
| `sktime/transformations` | Time series transformers (series, panel, hierarchical) |
| `sktime/classification` | Time series classifiers (including early classification) |
| `sktime/regression` | Time series regressors |
| `sktime/clustering` | Time series clusterers |
| `sktime/detection` | Anomaly / change-point detection |
| `sktime/distances` | Pairwise distance / kernel computations |
| `sktime/dists_kernels` | Distance and kernel estimators (BaseEstimator interface) |
| `sktime/pipeline` | Pipeline primitives and utilities |
| `sktime/networks` | Deep learning network architectures (Keras / PyTorch) |
| `sktime/performance_metrics` | Forecasting and detection error metrics |
| `sktime/benchmarking` | Benchmarking utilities and strategies |
| `sktime/split` | Cross-validation / data splitting utilities |
| `sktime/param_est` | Parameter estimators (e.g., seasonality) |
| `sktime/datasets` | Dataset loading utilities |
| `sktime/utils` | Shared utilities: testing, validation, datetime helpers, dependencies |
| `sktime/exceptions` | Custom exception classes |
| `sktime/catalogues` | Data catalogue / dataset index base classes |
| `extension_templates` | Cookiecutter-style templates for implementing new estimators |

### How modules interact

```
User code
    │
    ▼
sktime/[task]/[estimator].py   ◄──── imports base class
    │                                from sktime/[task]/base.py
    │ inherits
    ▼
sktime/[task]/base.py          ◄──── imports BaseEstimator
    │                                from sktime/base
    │ inherits
    ▼
sktime/base/_base.py           ─────► skbase.base (BaseObject, BaseEstimator)
    │
    ├──► sktime/datatypes        (input validation & conversion)
    ├──► sktime/registry         (tag lookup, estimator discovery)
    └──► sktime/utils            (dependencies, random state, etc.)
```

### Conceptual architecture diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER / APPLICATION                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  fit(), predict(), transform(), …
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│               LEARNING TASK MODULES  (public API layer)             │
│  forecasting │ classification │ regression │ clustering │ detection  │
│  transformations │ dists_kernels │ param_est │ alignment              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  inherits
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BASE CLASS LAYER                               │
│  BaseForecaster │ BaseTransformer │ BaseClassifier │ BaseClusterer  │
│  BaseRegressor  │ BaseAligner     │ BaseDistribution │ …             │
│                         (all inherit BaseEstimator)                 │
└────────────────────────┬────────────────────────────────────────────┘
                         │  uses
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌─────────────────┐
│  datatypes   │ │   registry   │ │      utils      │
│  (convert,   │ │ (all_estim., │ │ (deps, testing, │
│   validate)  │ │  tags, craft)│ │  random_state)  │
└──────────────┘ └──────────────┘ └─────────────────┘
```

---

## 3. Core Abstractions

### Base classes and abstract classes

| Class | Module | Description |
|-------|--------|-------------|
| `BaseObject` | `sktime.base._base` | Root of all sktime objects. Provides `get_params`/`set_params`, tag system, config system, `clone`, `reset`. |
| `BaseEstimator` | `sktime.base._base` | Adds `is_fitted`, `check_is_fitted`, `get_fitted_params`. Mixin of `TagAliaserMixin`, `skbase.BaseEstimator`, `BaseObject`. |
| `BasePanelMixin` | `sktime.base._base_panel` | Shared logic for panel data estimators (classifiers, regressors). |
| `BaseForecaster` | `sktime.forecasting.base` | Forecasting interface: `fit`, `predict`, `update`, `predict_interval`, `predict_quantiles`, `predict_proba`. |
| `BaseTransformer` | `sktime.transformations.base` | Transformation interface: `fit`, `transform`, `fit_transform`, `inverse_transform`. |
| `BaseClassifier` | `sktime.classification.base` | Panel classification: `fit`, `predict`, `predict_proba`. |
| `BaseEarlyClassifier` | `sktime.classification.early_classification.base` | Early classification with `update_predict`. |
| `BaseRegressor` | `sktime.regression.base` | Panel regression: `fit`, `predict`. |
| `BaseClusterer` | `sktime.clustering.base` | Clustering: `fit`, `predict`, `fit_predict`. |
| `BaseAligner` | `sktime.alignment.base` | Sequence alignment: `fit`, `get_alignment`. |
| `BasePairwiseTransformer` | `sktime.dists_kernels.base` | Pairwise distance/kernel on series. |
| `BasePairwiseTransformerPanel` | `sktime.dists_kernels.base` | Pairwise distance/kernel on panel data. |
| `BaseDistribution` | `sktime.base._proba._base` | Probabilistic output: `mean`, `var`, `pdf`, `cdf`, `ppf`, `sample`. |
| `BaseParamFitter` | `sktime.param_est.base` | Parameter estimator (fit parameters from data, not predictive). |
| `BaseDetector` | `sktime.detection.base` | Anomaly/change-point detection: `fit`, `predict`, `transform`. |
| `BaseDeepNetwork` | `sktime.networks.base` | Neural network architecture wrapper (no `fit`; yields a Keras model). |
| `BaseDeepClassifier` | `sktime.classification.deep_learning.base` | Deep classifier over Keras `BaseDeepNetwork`. |
| `BaseDeepRegressor` | `sktime.regression.deep_learning.base` | Deep regressor over Keras `BaseDeepNetwork`. |
| `BaseDeepNetworkPyTorch` | `sktime.forecasting.base.adapters._pytorch` | PyTorch-based forecasting adapter. |
| `BaseSplitter` | `sktime.split.base` | Cross-validation splitter yielding train/test index pairs. |
| `BaseMetric` | `sktime.performance_metrics.base` | Base for all performance metrics. |
| `BaseForecastingErrorMetric` | `sktime.performance_metrics.forecasting._base` | Forecasting error metrics with `evaluate` and `evaluate_by_index`. |
| `BaseDataset` | `sktime.datasets.base` | Abstract dataset loader. |
| `BaseDatatype` / `BaseConverter` | `sktime.datatypes._base` | Datatype definition and converter base. |

### Inheritance hierarchy (estimator-related)

```
skbase.BaseObject
└── sktime.BaseObject
    └── sktime.BaseEstimator
        ├── BaseForecaster           (_PredictProbaMixin mixin)
        │   ├── _BaseGlobalForecaster
        │   ├── BaseDeepNetworkPyTorch
        │   └── … (hundreds of forecasters)
        ├── BaseTransformer
        │   ├── _SeriesToSeriesTransformer     (legacy alias)
        │   ├── _SeriesToPrimitivesTransformer (legacy alias)
        │   ├── _PanelToTabularTransformer     (legacy alias)
        │   ├── _PanelToPanelTransformer       (legacy alias)
        │   └── … (hundreds of transformers)
        ├── BasePanelMixin
        │   ├── BaseClassifier
        │   │   ├── BaseEarlyClassifier
        │   │   ├── BaseDeepClassifier  (Keras)
        │   │   ├── BaseDeepClassifierPytorch
        │   │   └── … (many classifiers)
        │   └── BaseRegressor
        │       ├── BaseDeepRegressor
        │       └── … (many regressors)
        ├── BaseClusterer
        │   ├── BaseTimeSeriesLloyds
        │   └── … (clusterers)
        ├── BaseAligner
        ├── BasePairwiseTransformer
        ├── BasePairwiseTransformerPanel
        ├── BaseParamFitter
        └── BaseDetector

sktime.BaseObject (not BaseEstimator)
    ├── BaseDistribution
    ├── BaseDeepNetwork
    ├── BaseMetric
    │   └── BaseForecastingErrorMetric
    ├── BaseSplitter
    │   └── BaseWindowSplitter
    └── BaseDataset
```

### How polymorphism is used

Each scitype's base class defines a **public interface** (`fit`, `predict`, etc.) that delegates to **private underscore methods** (`_fit`, `_predict`, etc.). Subclasses override only the private methods, while the public methods contain all shared logic (input validation, type conversion, state management). This is the *Template Method* pattern.

### How extensibility is achieved

1. **Tag system** – estimators declare capabilities via `_tags` dict. Tags drive all conditional behavior in the base class (e.g., whether to call `_predict_quantiles`, whether to convert inputs).
2. **Inner type declaration** – `y_inner_mtype` / `X_inner_mtype` tags tell the base class which internal data format to convert to before calling `_fit`/`_predict`. Implementers never handle conversion.
3. **Extension templates** – `extension_templates/` provides ready-made implementations for each scitype.
4. **Registry auto-discovery** – `all_estimators()` scans the package and finds all concrete subclasses automatically; no manual registration needed.

---

## 4. Model Execution Flow

### Typical forecaster lifecycle

#### Initialization

```python
forecaster = MyForecaster(param1=value1)
```

- `__init__` stores parameters as instance attributes.
- `BaseObject.__init__` validates parameter names match the constructor signature.
- `_tags` class attribute merged with dynamic `_tags_dynamic` dict.
- `_check_estimator_deps` verifies soft dependencies declared in tags.

#### `fit(y, X=None, fh=None)`

```
BaseForecaster.fit(y, X, fh)
  1. validate_fh(fh)                     → stores self._fh
  2. check_is_scitype(y, "Series/Panel") → validates data format
  3. convert_to(y, self.get_tag("y_inner_mtype"))   → converts to internal mtype
  4. convert_to(X, self.get_tag("X_inner_mtype"))   → converts exogenous
  5. self._fit(y_inner, X_inner, fh)     → CALLS SUBCLASS IMPLEMENTATION
  6. self._is_fitted = True
  7. return self
```

#### `predict(fh=None, X=None)`

```
BaseForecaster.predict(fh, X)
  1. check_is_fitted()
  2. validate_fh(fh)
  3. convert X to inner mtype
  4. self._predict(fh_inner, X_inner)    → CALLS SUBCLASS IMPLEMENTATION
  5. convert output back to output mtype (pd.Series / pd.DataFrame)
  6. return predictions
```

#### Probabilistic prediction

```
BaseForecaster.predict_proba(fh, X, marginal)
  │
  ├── if tag "capability:pred_int" == True:
  │       calls self._predict_proba(fh, X)  → returns BaseDistribution
  └── else:
          falls back to Gaussian approximation from predict_interval
```

**Predictive distribution classes** (`sktime/base/_proba/`):
- `BaseDistribution` – abstract base; implements `mean`, `var`, `pdf`, `cdf`, `ppf`, `sample`, `energy`.
- `Normal` (`sktime/base/_proba/_normal.py`) – concrete Normal distribution.
- Other distributions are in `sktime/proba/` (e.g., `Laplace`, `LogNormal`, `TDistribution`, …).
- Distributions propagate through: `_predict_proba` → `BaseForecaster.predict_proba` → returned to user or consumed by `EvaluationMetric`.

### Modules involved step by step

```
1. sktime/[task]/[estimator].py          ← user entry point
2. sktime/[task]/base/_base.py           ← public interface, delegation
3. sktime/datatypes                      ← input validation & conversion
4. sktime/forecasting/base/_fh.py        ← ForecastingHorizon handling
5. sktime/[task]/[estimator].py          ← _fit, _predict (concrete logic)
6. sktime/base/_proba/                   ← distribution objects (if proba)
7. sktime/datatypes                      ← output conversion
```

---

## 5. Data Flow

### Full lifecycle of data

```
USER DATA (any supported format)
    │
    ▼  check_is_scitype()  – identify scitype (Series/Panel/Hierarchical)
    │
    ▼  check_is_mtype()    – validate internal consistency
    │
    ▼  convert_to(inner_mtype)  – convert to estimator's preferred format
    │         (handled transparently by base class)
    │
    ▼  [estimator]._fit(y_inner, X_inner)
    │         stores fitted state as attributes ending in "_"
    │
    ▼  [estimator]._predict(fh_inner, X_inner)
    │         produces output in inner format
    │
    ▼  convert output → output_mtype (usually pd.Series / pd.DataFrame)
    │
    ▼  USER RECEIVES OUTPUT
```

### Supported data formats (mtypes)

**Series** (univariate/multivariate single time series):
- `pd.Series` – univariate
- `pd.DataFrame` – multivariate
- `np.ndarray` – 1-D or 2-D

**Panel** (multiple time series):
- `pd.DataFrame` with `MultiIndex` (instance, time)
- `nested_univ` – `pd.DataFrame` with cells containing `pd.Series`
- `numpy3D` – `np.ndarray` of shape `(n_instances, n_columns, n_timepoints)`
- `df-list` – list of `pd.DataFrame`

**Hierarchical** (multi-level panel):
- `pd.DataFrame` with 3+ level `MultiIndex`

### VectorizedDF

`sktime.datatypes.VectorizedDF` is a special wrapper that enables estimators that only handle univariate series to be automatically applied in parallel across multivariate or panel datasets. The base class detects when vectorization is needed (via tags) and wraps data transparently.

---

## 6. Design Patterns

### Template Method

**Where**: Every base class (`BaseForecaster.fit` → `_fit`, `BaseTransformer.transform` → `_transform`, etc.)
**Why**: Separates the public API contract (validation, conversion, state management) from the algorithm-specific implementation. Contributors only override private methods.

### Strategy

**Where**: Estimator selection in pipelines and model selection classes (`ForecastingPipeline`, `GridSearchCV`, `MultiplexForecaster`).
**Why**: Allows swapping algorithm implementations at runtime without changing surrounding pipeline code.

### Adapter

**Where**: `sktime/forecasting/base/adapters/` (statsmodels, pmdarima, tbats, Prophet adapters); `sktime/classification/deep_learning/base/` (Keras/PyTorch adapters).
**Why**: Wraps third-party libraries behind the sktime interface without modifying the upstream code.

### Composite

**Where**: `TransformedTargetForecaster`, `ForecastingPipeline`, `ClassifierPipeline`, `ColumnEnsembleClassifier`, `MultiplexForecaster`.
**Why**: Allows building complex estimators from simpler ones that all share the same interface.

### Registry / Factory

**Where**: `sktime/registry/_craft.py` – `craft()` function; `all_estimators()` in `_lookup.py`.
**Why**: `craft(name_or_class, **params)` constructs estimators by name without imports. `all_estimators()` provides programmatic discovery for benchmarking and testing.

### Decorator

**Where**: `@classmethod get_test_params`, `@property is_fitted`, `@property fh`, `@property cutoff`.
**Why**: Encapsulates stateful checks (fitted state, forecasting horizon) as properties rather than method calls.

### Tag System (Capability Declaration)

**Where**: `_tags` class dict on every estimator; read by `get_tag()` / `get_class_tag()`.
**Why**: Tags act as a declarative capability declaration, driving conditional behavior in the base class without requiring subclasses to override methods for every feature. This is a form of *data-driven polymorphism*.

### Dependency Injection

**Where**: `sktime/utils/dependencies.py` – `_check_soft_dependencies`, `_check_estimator_deps`.
**Why**: Soft dependencies (e.g., statsforecast, pytorch, tensorflow) are checked at import or method call time rather than at the top of the file, avoiding hard failures when optional packages are not installed.

---

## 7. Dependency Structure

### Internal dependencies

```
sktime.base
    ├── depends on: skbase, sklearn (external)
    ├── depends on: sktime.utils, sktime.datatypes, sktime.exceptions
    └── imported by: ALL other sktime modules

sktime.datatypes
    ├── depends on: sktime.base (BaseObject), numpy, pandas
    └── imported by: all estimator base classes

sktime.registry
    ├── depends on: sktime.base, skbase
    └── imported by: sktime.tests, pipeline, utils

sktime.[task].base
    ├── depends on: sktime.base, sktime.datatypes, sktime.utils
    └── imported by: all concrete estimators in that task

sktime.utils
    ├── depends on: numpy, pandas, sktime.exceptions
    └── imported by: sktime.base, sktime.[task].base

sktime.pipeline
    ├── depends on: sktime.base, sktime.transformations, sktime.forecasting
    └── imported by: users, test suite
```

### Central modules (high fan-in)

| Module | Reason |
|--------|--------|
| `sktime.base._base` | Imported by every estimator and base class in the project |
| `sktime.datatypes` | Imported by every task base class for I/O conversion |
| `sktime.utils.dependencies` | Imported by almost every module for soft-dep checking |
| `sktime.registry` | Imported by tests, benchmarking, pipeline utilities |

---

## 8. Testing Framework

### Architecture – three layers

```
Layer 1 – Package level  (sktime/tests/test_all_estimators.py)
    Tests interface compliance with BaseObject / BaseEstimator specifications
    for ALL estimators in the registry.

Layer 2 – Module level   (sktime/[task]/tests/test_all_[task]s.py)
    Tests interface compliance with the task-specific base class.
    e.g., test_all_forecasters.py, test_all_classifiers.py

Layer 3 – Unit level     (sktime/[module]/tests/*.py)
    Tests individual algorithms, utilities, edge cases.
```

### How new components are validated

1. Implement estimator extending the correct base class.
2. Implement `get_test_params()` class method returning a list of parameter dicts.
3. Run `check_estimator(MyEstimator)` (wraps pytest test suite).
4. All Layer 1 and Layer 2 tests automatically pick up the new estimator via `all_estimators()`.
5. Optional: add a dedicated unit test file in `sktime/[task]/tests/`.

### Test design philosophy

- **Scenario-based**: `sktime/utils/_testing/scenarios_getter.py` retrieves canonical test scenarios per scitype.
- **Conditional execution**: `test_switch.py` and `run_test_for_class()` skip tests for estimators whose dependencies are unavailable or whose tags indicate skipping.
- **Subsampling**: `subsample_by_version_os` distributes test load across OS/Python-version matrix.
- **Non-state-changing methods**: a curated list of methods that must not alter estimator state is tested explicitly.
- **Reproducibility**: `set_random_state` is called on every estimator before testing.

### How estimators are verified for correctness

- `check_estimator` in `sktime.utils.estimator_checks` runs the full Layer 1+2 suite against a single estimator.
- Tests verify: `fit→predict` runs without error, output shapes match, `clone` produces identical output, serialisation round-trips work, no state is leaked between calls.

---

## 9. Active Development Zones

### Under active development

- **`sktime/detection/`** – Anomaly and change-point detection is the newest major learning task module; API is still evolving.
- **`sktime/base/_proba/`** – Probabilistic output / distribution framework is actively expanded with new distributions and capabilities.
- **`sktime/forecasting/`** – Global forecasting (`_BaseGlobalForecaster`), deep learning adapters, and hierarchical reconciliation are actively added.
- **`sktime/transformations/hierarchical/`** – Hierarchical series transformations.
- **`sktime/registry/_base_classes.py`** – Scitype registration system is still being formalised.
- **`sktime/networks/`** – New deep learning architectures (LTSF, ConvTimeNet) are being added.

### Experimental / early stage

- `sktime/catalogues/` – Data catalogue framework is minimal.
- `sktime/alignment/` – Small number of algorithms; API less mature.
- PyTorch adapters in forecasting and classification deep learning.

### Stable

- `sktime/base/_base.py` – Core `BaseObject`/`BaseEstimator` – highly stable.
- `sktime/datatypes/` – Conversion and validation infrastructure – stable.
- `sktime/forecasting/base/_base.py` – `BaseForecaster` – stable, well-tested.
- `sktime/transformations/base.py` – `BaseTransformer` – stable.
- `sktime/classification/base.py` – `BaseClassifier` – stable.
- `sktime/split/` – Splitting utilities – stable.
- `sktime/utils/` – Utility functions – mostly stable.

---

## 10. Complexity & Technical Debt

### Tightly coupled modules

- `sktime/forecasting/base/_base.py` is ~3000 lines and touches `datatypes`, `utils`, `base._proba`, `registry`, and `ForecastingHorizon` all at once. Changes here have wide blast radius.
- `sktime/transformations/base.py` similarly large and handles multiple scitypes (Series, Panel, Hierarchical) with complex dispatch logic.

### Large or complex files

| File | Reason for complexity |
|------|-----------------------|
| `sktime/forecasting/base/_base.py` | Central orchestration for all forecaster I/O, quantile/interval/proba prediction dispatch, vectorization |
| `sktime/transformations/base.py` | Handles Series, Panel, and Hierarchical transformations with heavy mtype dispatch |
| `sktime/datatypes/_convert.py` | Large conversion matrix across all mtypes |
| `sktime/tests/test_all_estimators.py` | Comprehensive interface compliance tests for ALL estimators |
| `sktime/utils/_testing/estimator_checks.py` | Deep estimator checking utilities |

### Unclear abstractions

- The `series_as_features` module is a legacy naming – its role is largely superseded by `panel` handling in transformations.
- `_SeriesToSeriesTransformer`, `_SeriesToPrimitivesTransformer`, etc., in `transformations/base.py` are legacy stub classes kept for backward compatibility.
- `BasePanelMixin` adds shared panel logic but its boundary with `BaseClassifier`/`BaseRegressor` is sometimes unclear.

### Potential design inconsistencies

- Deep learning networks in `sktime/networks/` inherit `BaseDeepNetwork(BaseObject)` (not `BaseEstimator`) – they are architecture blueprints, not fittable estimators. The distinction is correct but subtle.
- `BaseDetector` has a `transform` method (returns a binary/labeled series) *and* a `predict` method, which can cause confusion about the canonical output method.
- Some modules have both `base.py` (single file) and `base/` (directory) patterns.

---

## 11. Performance Analysis

### Computational bottlenecks

- **Data conversion** (`sktime/datatypes/_convert.py`): Every `fit`/`predict` call may convert data between mtypes. This is O(n) in data size and can be significant for large panel datasets.
- **VectorizedDF**: Broadcasting univariate estimators across panel/hierarchical data uses `pandas` groupby + apply, which is Python-level looping and not vectorized.
- **Deep learning**: `BaseDeepClassifier` and `BaseDeepNetworkPyTorch` call into Keras/PyTorch; their performance depends entirely on the backend.

### Memory-heavy components

- `nested_univ` mtype (panel as nested pandas DataFrame) stores `pd.Series` objects in DataFrame cells – high memory overhead for large datasets.
- `numpy3D` mtype is the most memory-efficient for equal-length panel data.
- `VectorizedDF` keeps both the original and vectorized views in memory simultaneously.

### Scalability risks

- `all_estimators()` scans the entire package at import time (cached after first call); this can be slow on first run with many optional dependencies.
- `test_all_estimators.py` instantiates *every* estimator; test runs can be slow without subsampling/parallelization.

### Opportunities for vectorization / GPU

- Replacing `nested_univ` with `numpy3D` or `pd.DataFrame` with MultiIndex enables numpy vectorization.
- The `backend:parallel` config on `BaseForecaster` supports `dask`, `ray`, `loky`, `multiprocessing` for broadcasting forecasters across panel instances.
- GPU: only via deep learning adapters (Keras/PyTorch); pure Python algorithms have no GPU path.

---

## 12. Extension Mechanisms

### Adding a new forecaster

**1. File location**:
```
sktime/forecasting/[your_module_name]/_[your_estimator].py
```

**2. Base class to extend**:
```python
from sktime.forecasting.base import BaseForecaster
class MyForecaster(BaseForecaster): ...
```

**3. Methods that MUST be implemented**:
```python
def _fit(self, y, X=None, fh=None): ...
def _predict(self, fh, X=None): ...
```

**4. Methods optionally implemented** (enabled via tags):
```python
def _predict_quantiles(self, fh, X=None, alpha=None): ...
def _predict_interval(self, fh, X=None, coverage=None): ...
def _predict_var(self, fh, X=None, cov=False): ...
def _predict_proba(self, fh, X=None): ...          # returns BaseDistribution
def _update(self, y, X=None, update_params=True): ...
def _get_fitted_params(self): ...
```

**5. Declare capabilities via tags**:
```python
_tags = {
    "y_inner_mtype": "pd.Series",           # internal data format
    "X_inner_mtype": "pd.DataFrame",        # exogenous data format
    "scitype:y": "univariate",              # "univariate", "multivariate", "both"
    "capability:pred_int": True,            # set True if intervals implemented
    "requires-fh-in-fit": False,            # set False if fh not needed in fit
    "python_dependencies": None,            # e.g., ["statsmodels>=0.12"]
}
```

**6. Required for testing** – implement `get_test_params`:
```python
@classmethod
def get_test_params(cls, parameter_set="default"):
    return [{"param1": 1}, {"param1": 2, "param2": "other"}]
```

**7. Registration** – automatic via module `__init__.py`:
Add the class to the `__init__.py` of the containing module. `all_estimators()` will discover it automatically by crawling the package.

**8. Testing**:
```python
from sktime.utils.estimator_checks import check_estimator
check_estimator(MyForecaster)
```

### Adding a new probabilistic model / distribution

**File location**: `sktime/proba/[your_distribution].py`

**Base class**: `sktime.base._proba._base.BaseDistribution`

**Methods to implement** (at minimum):
```python
def _mean(self): ...    # analytical mean
def _var(self): ...     # analytical variance
def _pdf(self, x): ...  # probability density function
def _cdf(self, x): ...  # cumulative distribution function
```

Optional (approximate methods with Monte Carlo fallback in base class):
```python
def _ppf(self, p): ...           # quantile / percent-point function
def _energy_self(self): ...      # energy distance to itself
def _sample(self, n_samples): ...
```

**Tags**:
```python
_tags = {
    "capabilities:approx": [],   # list methods that need MC approximation
    "python_dependencies": None,
}
```

### Adding a new deep learning model

**File location**: `sktime/networks/[your_architecture].py` (architecture only), then
`sktime/forecasting/deep_learning/[your_forecaster].py` or
`sktime/classification/deep_learning/[your_classifier].py`

**Base class for architecture** (Keras):
```python
from sktime.networks.base import BaseDeepNetwork
class MyNetwork(BaseDeepNetwork):
    def build_network(self, input_shape, **kwargs):
        # return a keras.Model
```

**Base class for classifier** (Keras):
```python
from sktime.classification.deep_learning.base import BaseDeepClassifier
class MyDeepClassifier(BaseDeepClassifier):
    def build_model(self, input_shape, n_classes, **kwargs): ...
```

**Base class for forecaster** (PyTorch):
```python
from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
class MyPyTorchForecaster(BaseDeepNetworkPyTorch):
    def _build_network(self, fh, X): ...
```

---

## 13. Contribution Entry Points

### Beginner level

- **Fix docstrings / type hints** – most modules have incomplete docstrings, especially parameter descriptions. No algorithmic knowledge required.
- **Add dataset loaders** – `sktime/datasets/` has a clear `BaseDataset` template and many simple loaders to follow.
- **Fix deprecation warnings** – search for `FutureWarning` or `DeprecationWarning` throughout the codebase; these often need small migration patches.
- **Add test cases** (`get_test_params`)  – many estimators have only one test parameter set; adding diverse cases improves coverage.
- **Report / fix small bugs** – the issue tracker lists many `bug` labeled issues that are self-contained.

### Intermediate level

- **Wrap a new forecasting library** – use `sktime/forecasting/base/adapters/` as a pattern; wrap statsforecast, neuralforecast, or other libraries behind the `BaseForecaster` interface.
- **Add a new transformer** – `extension_templates/transformer.py` provides the complete template; contributing a new series transformer is a well-defined task.
- **Add a probabilistic distribution** – follow `sktime/proba/` pattern; contribute a new distribution subclassing `BaseDistribution`.
- **Improve type conversion** – `sktime/datatypes/` always needs more efficient or more robust converters between mtypes.
- **Add metrics** – `sktime/performance_metrics/` has clear `BaseForecastingErrorMetric` templates.

### Advanced architecture level

- **Improve `VectorizedDF`** – the current pandas-based broadcasting is slow for large datasets; a vectorized or dask-backed replacement would have major impact.
- **Refactor `BaseForecaster`** – the `_base.py` file is very large; a clean decomposition into mixins (proba mixin, vectorization mixin, update mixin) would improve maintainability.
- **Extend `detection` module** – the `BaseDetector` API is still young; contributing new algorithms and ironing out the API is high-impact.
- **Improve CI/CD and testing infrastructure** – `test_all_estimators.py` is complex; contributing better subsampling, parallelization, or caching would speed up CI.
- **Hierarchical forecasting** – `sktime/transformations/hierarchical/` and `_BaseGlobalForecaster` are actively evolving; contributing reconciliation algorithms or global model wrappers is valuable.

---

## 14. Roadmap Signals

### Long-term evolution implied by code structure

- **Unified probabilistic forecasting**: The `BaseDistribution` → `predict_proba` pipeline is already built end-to-end; ongoing work is expanding the distribution library and making probabilistic evaluation metrics first-class.
- **Global/panel forecasting**: `_BaseGlobalForecaster` is explicitly marked as a "temporary solution, might be merged into BaseForecaster later" – signaling a planned unification of local vs. global forecaster interfaces.
- **Scitype formalisation**: `sktime/registry/_base_classes.py` introduces a new metaclass-like `_BaseScitypeOfObject` system for formally registering scitypes; this replaces the older hand-maintained register and implies more dynamic extensibility.
- **Deeper PyTorch integration**: The `BaseDeepNetworkPyTorch` adapter in forecasting and `BaseDeepClassifierPytorch` in classification suggest a planned parity between Keras and PyTorch support.
- **Detection as a first-class task**: The `detection` module is growing rapidly; a full benchmarking suite and more algorithms are implied.

### Architectural transitions happening

- **skbase extraction**: Core `BaseObject`/`BaseEstimator` logic has been extracted into the separate `skbase` package; sktime now inherits from `skbase` rather than defining these classes itself. This enables other libraries to adopt the same base without depending on sktime.
- **Registry refactor**: The old `REGISTRY` dict is being replaced by the dynamic `_BaseScitypeOfObject` class-based system.
- **Deprecation of legacy mtype aliases**: `_SeriesToSeriesTransformer` and related legacy classes are kept for backward compatibility but are no longer the recommended pattern.

---

## 15. Strategic Summary

### Main strengths of the codebase

1. **Unified interface design** – The `fit/predict` contract, tag system, and automatic type conversion mean that users can swap algorithms with a single line change, and contributors can implement new algorithms without boilerplate.
2. **Breadth** – Hundreds of estimators across forecasting, classification, regression, clustering, detection, and transformation make sktime the most comprehensive time series ML library in Python.
3. **Scikit-learn interoperability** – Seamless integration with the sklearn ecosystem lowers the barrier to adoption.
4. **Mature testing framework** – The three-layer test architecture means every new estimator is automatically tested against the full interface contract.
5. **Extension templates** – `extension_templates/` makes it straightforward to contribute correct, interface-compliant estimators even without deep knowledge of the internals.

### Biggest architectural challenges

1. **`BaseForecaster` and `BaseTransformer` size** – These files are load-bearing and very large; any significant change requires deep understanding of the entire mtype/dispatch logic.
2. **Data conversion overhead** – The mtype conversion system, while powerful, adds latency on every `fit`/`predict` call. For production workloads this can be significant.
3. **Deep learning fragmentation** – Keras and PyTorch adapters have separate base classes with partially duplicated logic; a unified deep learning interface would reduce maintenance burden.
4. **Soft dependency management** – With hundreds of optional dependencies, ensuring CI does not break when a dependency is absent requires constant attention.
5. **Scitype boundary evolution** – As new scitypes (detection, hierarchical, global forecasting) are added, the boundary conditions between them (what data format, what methods, what tags) need careful design to remain consistent.

### Most impactful areas for contribution

1. **Detection module** – Still young; new algorithms, API refinements, and test scenarios have high leverage.
2. **Probabilistic forecasting** – Adding distributions, improving `predict_proba` integration, and expanding evaluation metrics.
3. **Global/hierarchical forecasting** – Active frontier; `_BaseGlobalForecaster` and reconciliation transformers.
4. **Performance / scalability** – Replacing pandas-level loops in `VectorizedDF` with vectorized or distributed alternatives.
5. **Documentation and tutorials** – Improving end-to-end tutorials for each learning task, especially the newer ones.

### Fastest way to become a core contributor

1. **Read `extension_templates/forecasting.py`** (and other templates) thoroughly – they are the most concise specification of what every estimator must do.
2. **Run `check_estimator`** on a new estimator you implement – this exercises the full interface compliance suite and teaches you exactly what the framework expects.
3. **Read `BaseForecaster.fit` and `BaseForecaster.predict`** in `sktime/forecasting/base/_base.py` – understanding the delegation pattern here gives you a complete mental model of how every estimator type works.
4. **Contribute a small, self-contained estimator** (a new transformer or a library wrapper) – this forces engagement with the tag system, `get_test_params`, and the PR review process.
5. **Engage with the issue tracker** – filtering by `good first issue` or `help wanted` in the GitHub repository surfaces tasks where guidance and mentorship are available.
