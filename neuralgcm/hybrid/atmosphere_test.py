# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for how the hybrid atmosphere applies parameterization tendencies.

Most tests start from an atmosphere at rest over flat orography. The dycore
leaves that state unchanged, and a horizontally uniform temperature tendency
keeps it at rest, so temperature must change by exactly the time integral of the
tendency that `learned_physics` returned. This gives closed-form expectations
for when `learned_physics` is called, what it and each diagnostic stage see, and
how fixers modify its tendencies.
"""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.base import parameterizations
from neuralgcm.classical import primitive_equations
from neuralgcm.hybrid import atmosphere as hybrid_atmosphere
import numpy as np
from terrax.atmosphere import equations as atm_equations
from terrax.atmosphere import fixers
from terrax.core import coordinates
from terrax.core import diagnostics
from terrax.core import dynamic_io
from terrax.core import orographies
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import time_integrators
from terrax.core import units

PROGNOSTICS_KEY = 'atmosphere_prognostics'
NONDIM_PROGNOSTICS_KEY = 'nondim_atmosphere_prognostics'
FORCING_KEY = 'forcing'
# Heating rates in K/s on each of the four model levels. Distinct values make
# sure that tendencies are not mixed between levels.
HEATING_RATES_SI = (1e-4, 2e-4, 3e-4, 4e-4)
ONE_HOUR = np.timedelta64(1, 'h')
HALF_HOUR = np.timedelta64(30, 'm')
QUARTER_HOUR = np.timedelta64(15, 'm')


def _seconds(timedelta: np.timedelta64) -> float:
  return float(timedelta / np.timedelta64(1, 's'))


def _zero_tendencies(inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
  return {k: 0.0 * v for k, v in inputs.items() if k != 'time'}


class _ColumnHeating(nnx.Module):
  """Physics that heats every column at a prescribed rate on each level.

  The temperature tendency is `scale * rates`, where `scale` is 1 by default,
  the number of seconds since `ramp_start` if it is set, or the value of
  `forcing['heating_scale']` at the current time if `forcing` is set. All other
  tendencies are zero.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels,
      rates: np.ndarray,
      ramp_start: jdt.Datetime | None = None,
      forcing: dynamic_io.DynamicInputModule | None = None,
  ):
    self.ylm_map = ylm_map
    self.levels = levels
    self.rates = nnx.Param(jnp.asarray(rates, dtype=jnp.float32))
    self.ramp_start = nnx.data(ramp_start)
    self.forcing = forcing

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    grid = self.ylm_map.lon_lat_grid
    scale = 1.0
    if self.ramp_start is not None:
      scale = (inputs['time'].data - self.ramp_start).total_seconds()
    if self.forcing is not None:
      scale = self.forcing(inputs['time'])['heating_scale'].data
    heating = (
        cx.field(scale * self.rates[...], self.levels)
        * cx.field(jnp.ones(grid.shape), grid)
    )
    tendencies = _zero_tendencies(inputs)
    tendencies['temperature'] = self.ylm_map.to_modal(heating)
    return tendencies


class _ConstantNodalHeating(nnx.Module):
  """Nodal tendency transform for `ModalParameterization`."""

  def __init__(
      self,
      grid: coordinates.LonLatGrid,
      levels: coordinates.SigmaLevels,
      rates: np.ndarray,
  ):
    self.grid = grid
    self.levels = levels
    self.rates = rates

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    heating = (
        cx.field(jnp.asarray(self.rates, dtype=jnp.float32), self.levels)
        * cx.field(jnp.ones(self.grid.shape), self.grid)
    )
    return {'temperature': heating}


class _ExtractTemperature(nnx.Module):
  """Extracts `temperature` from diagnosed inputs or from `prognostics`."""

  def __init__(self, name: str, from_prognostics: bool):
    self.name = name
    self.from_prognostics = from_prognostics

  def __call__(self, inputs, *args, **kwargs) -> dict[str, cx.Field]:
    del args  # unused.
    source = kwargs['prognostics'] if self.from_prognostics else inputs
    return {self.name: source['temperature']}


class _ExtractConstant(nnx.Module):
  """Extracts a constant field that does not depend on diagnosed inputs."""

  def __init__(self, name: str, value: cx.Field):
    self.name = name
    self.value = nnx.data(value)

  def __call__(self, inputs, *args, **kwargs) -> dict[str, cx.Field]:
    del inputs, args, kwargs  # unused.
    return {self.name: self.value}


class _ScaleTemperature(nnx.Module):
  """Fixer that multiplies `temperature` by a scalar diagnostic."""

  def __init__(self, diagnostic_key: str):
    self.diagnostic_key = diagnostic_key

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    return {self.diagnostic_key: cx.Scalar()}

  def __call__(self, diagnostic, targets, *args, **kwargs):
    del args, kwargs  # unused.
    scaled = targets['temperature'] * diagnostic[self.diagnostic_key]
    return targets | {'temperature': scaled}


class _AddToTemperature(nnx.Module):
  """Fixer that adds a diagnostic on the modal volume to `temperature`."""

  def __init__(self, diagnostic_key: str, coord: cx.Coordinate):
    self.diagnostic_key = diagnostic_key
    self.coord = coord

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    return {self.diagnostic_key: self.coord}

  def __call__(self, diagnostic, targets, *args, **kwargs):
    del args, kwargs  # unused.
    shifted = targets['temperature'] + diagnostic[self.diagnostic_key]
    return targets | {'temperature': shifted}


class _SetTemperature(nnx.Module):
  """Fixer that replaces `temperature` by a diagnostic on the modal volume."""

  def __init__(self, diagnostic_key: str, coord: cx.Coordinate):
    self.diagnostic_key = diagnostic_key
    self.coord = coord

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    return {self.diagnostic_key: self.coord}

  def __call__(self, diagnostic, targets, *args, **kwargs):
    del args, kwargs  # unused.
    return targets | {'temperature': diagnostic[self.diagnostic_key]}


class LearnedParameterizationSpectralDycoreTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(len(HEATING_RATES_SI))
    self.nodal_volume = cx.coords.compose(self.levels, self.grid)
    self.modal_volume = cx.coords.compose(self.levels, self.ylm_grid)
    self.init_time = jdt.to_datetime('2020-01-01T00:00')

  def _ylm_map(
      self, sim_units: units.SimUnits
  ) -> spherical_harmonics.FixedYlmMapping:
    return spherical_harmonics.FixedYlmMapping(
        self.grid, self.ylm_grid, radius=sim_units.radius
    )

  def _rates(self, sim_units: units.SimUnits) -> np.ndarray:
    """Returns nondimensional heating rates, i.e. K per unit of sim time."""
    one_second = sim_units.nondimensionalize_timedelta64(np.timedelta64(1, 's'))
    return np.array(HEATING_RATES_SI) / one_second

  def _heating(
      self, sim_units: units.SimUnits = units.SI_UNITS, **kwargs
  ) -> _ColumnHeating:
    return _ColumnHeating(
        self._ylm_map(sim_units), self.levels, self._rates(sim_units), **kwargs
    )

  def _modal_filter(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      sim_units: units.SimUnits,
      model_timestep: np.timedelta64,
  ) -> spatial_filters.ExponentialModalFilter:
    return spatial_filters.ExponentialModalFilter.from_timescale(
        ylm_map=ylm_map,
        dt=sim_units.nondimensionalize_timedelta64(model_timestep),
        timescale='2 hours',
        order=4,
        cutoff=0.0,
        sim_units=sim_units,
    )

  def _make_model(
      self,
      learned_physics,
      sim_units: units.SimUnits = units.SI_UNITS,
      model_timestep: np.timedelta64 = ONE_HOUR,
      **kwargs,
  ) -> hybrid_atmosphere.LearnedParameterizationSpectralDycore:
    ylm_map = self._ylm_map(sim_units)
    return hybrid_atmosphere.LearnedParameterizationSpectralDycore(
        model_ylm_map=ylm_map,
        model_levels=self.levels,
        model_orography=orographies.ModalOrography(
            ylm_map=ylm_map, rngs=nnx.Rngs(0)
        ),
        model_timestep=model_timestep,
        modal_filter=self._modal_filter(ylm_map, sim_units, model_timestep),
        learned_physics=learned_physics,
        sim_units=sim_units,
        **kwargs,
    )

  def _resting_state(self) -> dict[str, cx.Field]:
    """Returns nondimensional prognostics of an atmosphere at rest."""
    zeros = cx.field(jnp.zeros(self.modal_volume.shape), self.modal_volume)
    return {
        'divergence': zeros,
        'vorticity': zeros,
        'temperature': zeros,
        'specific_humidity': zeros,
        'log_surface_pressure': cx.field(
            jnp.zeros(self.ylm_grid.shape), self.ylm_grid
        ),
        'time': cx.field(self.init_time),
    }

  def _advance_from_rest(self, model, steps: int = 1) -> None:
    model.assimilate({NONDIM_PROGNOSTICS_KEY: self._resting_state()})
    for _ in range(steps):
      model.advance()

  def _observe_temperature(self, model) -> np.ndarray:
    """Returns temperature in Kelvin with dimensions (level, lon, lat)."""
    observed = model.observe(
        {PROGNOSTICS_KEY: {'temperature': self.nodal_volume}}
    )
    return np.asarray(observed[PROGNOSTICS_KEY]['temperature'].data)

  def _to_nodal(self, model, modal: cx.Field) -> np.ndarray:
    return np.asarray(model.model_ylm_map.to_nodal(modal).data)

  def _assert_uniform_on_levels(
      self, actual: np.ndarray, expected_on_levels: np.ndarray, rtol=1e-4
  ) -> None:
    """Asserts that `actual` equals `expected_on_levels` in every column."""
    expected = np.broadcast_to(
        np.asarray(expected_on_levels)[:, np.newaxis, np.newaxis], actual.shape
    )
    atol = rtol * np.abs(expected_on_levels).max()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

  def _assert_at_rest(self, model) -> None:
    """Asserts that everything other than temperature and time is unchanged."""
    for k, v in model.prognostic_values().items():
      if k not in ('temperature', 'time'):
        np.testing.assert_allclose(v.data, 0.0, atol=1e-5, err_msg=k)

  @parameterized.named_parameters(
      dict(testcase_name='defaults', model_kwargs={}),
      dict(
          testcase_name='parameterization_substeps',
          model_kwargs=dict(parameterization_timestep=QUARTER_HOUR),
      ),
      dict(
          testcase_name='aggregation_and_parameterization_substeps',
          model_kwargs=dict(
              aggregate_every=HALF_HOUR,
              parameterization_timestep=QUARTER_HOUR,
          ),
      ),
      dict(
          testcase_name='dycore_substeps',
          model_kwargs=dict(dycore_substeps=3),
      ),
      dict(
          testcase_name='semi_lagrangian',
          model_kwargs=dict(
              dycore_equation_cls=(
                  atm_equations.SemiLagrangianPrimitiveEquations
              ),
              time_integrator_cls=(
                  time_integrators.SemiLagrangianCrankNicolsonRK2
              ),
          ),
      ),
      dict(
          testcase_name='nondimensional_sim_units',
          model_kwargs={},
          sim_units=units.DEFAULT_UNITS,
      ),
  )
  def test_constant_heating_integrates_over_model_timestep(
      self, model_kwargs, sim_units=units.SI_UNITS
  ):
    model = self._make_model(
        self._heating(sim_units), sim_units=sim_units, **model_kwargs
    )
    self._advance_from_rest(model, steps=2)

    expected = np.array(HEATING_RATES_SI) * _seconds(2 * ONE_HOUR)
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)
    self._assert_at_rest(model)
    observed_time = model.observe({PROGNOSTICS_KEY: {'time': cx.Scalar()}})
    self.assertEqual(
        observed_time[PROGNOSTICS_KEY]['time'].data.to_datetime64(),
        self.init_time.to_datetime64() + 2 * ONE_HOUR,
    )

  def test_dual_grid_regrids_tendencies_to_dycore_grid(self):
    sim_units = units.SI_UNITS
    dycore_ylm_map = spherical_harmonics.FixedYlmMapping(
        coordinates.LonLatGrid.T31(),
        coordinates.SphericalHarmonicGrid.T31(),
        radius=sim_units.radius,
    )
    model = self._make_model(
        self._heating(),
        dycore_ylm_map=dycore_ylm_map,
        dycore_orography=orographies.ModalOrography(
            ylm_map=dycore_ylm_map, rngs=nnx.Rngs(0)
        ),
    )
    self._advance_from_rest(model)

    expected = np.array(HEATING_RATES_SI) * _seconds(ONE_HOUR)
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)
    self._assert_at_rest(model)

  def test_modal_parameterization_applies_nodal_heating(self):
    sim_units = units.SI_UNITS
    ylm_map = self._ylm_map(sim_units)
    learned_physics = parameterizations.ModalParameterization(
        ylm_map=ylm_map,
        nodal_tendency_transform=_ConstantNodalHeating(
            self.grid, self.levels, self._rates(sim_units)
        ),
        modal_filter=self._modal_filter(ylm_map, sim_units, ONE_HOUR),
    )
    model = self._make_model(learned_physics)
    self._advance_from_rest(model)

    expected = np.array(HEATING_RATES_SI) * _seconds(ONE_HOUR)
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)
    self._assert_at_rest(model)

  @parameterized.named_parameters(
      dict(
          testcase_name='one_evaluation',
          model_kwargs={},
          evaluation_times=(3600,),
          held_for=ONE_HOUR,
      ),
      dict(
          testcase_name='dycore_substeps_do_not_reevaluate',
          model_kwargs=dict(dycore_substeps=4),
          evaluation_times=(3600,),
          held_for=ONE_HOUR,
      ),
      dict(
          testcase_name='parameterization_substeps',
          model_kwargs=dict(parameterization_timestep=QUARTER_HOUR),
          evaluation_times=(3600, 4500, 5400, 6300),
          held_for=QUARTER_HOUR,
      ),
      dict(
          testcase_name='aggregation_and_parameterization_substeps',
          model_kwargs=dict(
              aggregate_every=HALF_HOUR,
              parameterization_timestep=QUARTER_HOUR,
              dycore_substeps=2,
          ),
          evaluation_times=(3600, 4500, 5400, 6300),
          held_for=QUARTER_HOUR,
      ),
  )
  def test_physics_is_evaluated_at_start_of_each_parameterization_timestep(
      self, model_kwargs, evaluation_times, held_for
  ):
    # The heating rate grows linearly with the time passed to `learned_physics`,
    # so the accumulated heating identifies the times at which it was called and
    # how long each returned tendency was applied for. `evaluation_times` are
    # in seconds since the start of the ramp, one hour before the initial time.
    ramp_start = self.init_time - jdt.to_timedelta(ONE_HOUR)
    model = self._make_model(
        self._heating(ramp_start=ramp_start), **model_kwargs
    )
    self._advance_from_rest(model)

    expected = (
        np.array(HEATING_RATES_SI) * sum(evaluation_times) * _seconds(held_for)
    )
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)

  def test_physics_reads_dynamic_inputs_at_current_time(self):
    forcing = dynamic_io.DynamicInputSlice(
        keys_to_coords={'heating_scale': cx.Scalar()},
        observation_key=FORCING_KEY,
    )
    model = self._make_model(
        self._heating(forcing=forcing),
        parameterization_timestep=QUARTER_HOUR,
    )
    timedelta = coordinates.TimeDelta(np.array([0, 30], 'timedelta64[m]'))
    forcing_times = self.init_time + jdt.to_timedelta(timedelta.deltas)
    model.update_dynamic_inputs({
        FORCING_KEY: {
            'heating_scale': cx.field(jnp.array([1.0, 3.0]), timedelta),
            'time': cx.field(forcing_times, timedelta),
        }
    })
    self._advance_from_rest(model)

    # The most recent forcing slice is used: 1 at 0 and 15 minutes, 3 at 30 and
    # 45 minutes.
    expected = np.array(HEATING_RATES_SI) * (1 + 1 + 3 + 3) * 900
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)

  def test_zero_tendencies_match_classical_dycore(self):
    sim_units = units.SI_UNITS
    ylm_map = self._ylm_map(sim_units)
    model = self._make_model(
        _zero_tendencies,
        parameterization_timestep=HALF_HOUR,
        dycore_substeps=3,
    )
    # The hybrid model takes 6 dycore steps per `model_timestep`.
    classical = primitive_equations.SpectralPrimitiveEquationsModel(
        ylm_map=ylm_map,
        levels=self.levels,
        model_timestep=ONE_HOUR,
        orography=orographies.ModalOrography(ylm_map=ylm_map, rngs=nnx.Rngs(0)),
        time_integrator_cls=time_integrators.ImexRk3Sil,
        modal_filter=self._modal_filter(ylm_map, sim_units, ONE_HOUR),
        sim_units=sim_units,
        num_substeps=6,
        tracer_names=('specific_humidity',),
    )
    longitude, latitude = (
        np.deg2rad(self.grid.fields[k].data)
        for k in ('longitude', 'latitude')
    )
    wave = np.cos(latitude) ** 2 * np.sin(2 * longitude[:, np.newaxis])
    sigma = self.levels.fields['sigma'].data[:, np.newaxis, np.newaxis]
    to_modal = lambda x: ylm_map.to_modal(cx.field(x, self.nodal_volume))
    state = self._resting_state() | {
        'temperature': to_modal(2.0 * sigma * wave),
        'vorticity': to_modal(1e-6 * sigma * wave),
        'specific_humidity': to_modal(1e-3 * (1 + sigma * wave)),
    }
    model.assimilate({NONDIM_PROGNOSTICS_KEY: state})
    classical.assimilate({'nondim_prognostics': state})
    model.advance()
    classical.advance()

    actual = model.prognostic_values()
    expected = classical.prognostic_values()
    self.assertSameElements(actual, expected)
    for k in ('divergence', 'vorticity', 'temperature', 'specific_humidity'):
      with self.subTest(k):
        scale = np.abs(expected[k].data).max()
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(
            actual[k].data, expected[k].data, atol=1e-4 * scale
        )
    # The state must have evolved for the comparison to be meaningful.
    self.assertGreater(np.abs(actual['divergence'].data).max(), 0.0)

  def test_diagnostic_stages_see_documented_inputs(self):
    def make(name: str, from_prognostics: bool):
      return diagnostics.CumulativeDiagnostic(
          extract=_ExtractTemperature(name, from_prognostics),
          extract_coords={name: self.modal_volume},
      )

    names = (
        'at_step_start',
        'tendency',
        'at_tendency_evaluation',
        'at_step_end',
        'at_aggregation',
    )
    model = self._make_model(
        self._heating(),
        aggregate_every=HALF_HOUR,
        parameterization_timestep=QUARTER_HOUR,
        step_start_diagnostics={'d': make('at_step_start', False)},
        tendency_diagnostics={
            'tendency': make('tendency', False),
            'prognostics': make('at_tendency_evaluation', True),
        },
        step_end_diagnostics={'d': make('at_step_end', False)},
        aggregation_diagnostics={'d': make('at_aggregation', False)},
    )
    self._advance_from_rest(model)
    observed = model.observe(
        {'nondim_diagnostics': {k: self.modal_volume for k in names}}
    )['nondim_diagnostics']

    rates = np.array(HEATING_RATES_SI)
    # Temperature is `rates * t`, and each diagnostic sums the values it saw.
    expected_sums = {
        # Start of each parameterization step: t = 0, 15, 30, 45 minutes.
        'at_step_start': rates * (0 + 900 + 1800 + 2700),
        # Tendencies of the four parameterization steps.
        'tendency': rates * 4,
        # Tendency diagnostics receive prognostics from the start of the step.
        'at_tendency_evaluation': rates * (0 + 900 + 1800 + 2700),
        # End of each parameterization step: t = 15, 30, 45, 60 minutes.
        'at_step_end': rates * (900 + 1800 + 2700 + 3600),
        # Every `aggregate_every`: t = 30, 60 minutes.
        'at_aggregation': rates * (1800 + 3600),
    }
    for name, expected in expected_sums.items():
      with self.subTest(name):
        self._assert_uniform_on_levels(
            self._to_nodal(model, observed[name]), expected
        )

  def test_interval_diagnostic_accumulates_tendencies(self):
    heating = diagnostics.IntervalDiagnostic(
        extract=_ExtractTemperature('heating', from_prognostics=False),
        extract_coords={'heating': self.modal_volume},
        interval=ONE_HOUR,
        resolution=HALF_HOUR,
    )
    model = self._make_model(
        self._heating(ramp_start=self.init_time),
        parameterization_timestep=QUARTER_HOUR,
        tendency_diagnostics={'heating': heating},
    )
    self._advance_from_rest(model, steps=2)
    observed = model.observe(
        {'nondim_diagnostics': {'heating': self.modal_volume}}
    )['nondim_diagnostics']

    # Sum of tendencies over the last hour, i.e. the second model step, in
    # which `learned_physics` is called at t = 60, 75, 90, 105 minutes.
    expected = np.array(HEATING_RATES_SI) * (3600 + 4500 + 5400 + 6300)
    self._assert_uniform_on_levels(
        self._to_nodal(model, observed['heating']), expected
    )

  def test_tendency_fixers_modify_tendencies_in_order(self):
    sim_units = units.SI_UNITS
    ylm_map = self._ylm_map(sim_units)
    offset_rate = 5e-4
    offset = ylm_map.to_modal(
        cx.field(jnp.full(self.nodal_volume.shape, offset_rate),
                 self.nodal_volume)
    )
    model = self._make_model(
        self._heating(),
        tendency_diagnostics={
            'scale': diagnostics.InstantDiagnostic(
                extract=_ExtractConstant('scale', cx.field(3.0)),
                extract_coords={'scale': cx.Scalar()},
            ),
            'offset': diagnostics.InstantDiagnostic(
                extract=_ExtractConstant('offset', offset),
                extract_coords={'offset': self.modal_volume},
            ),
        },
        tendency_fixers=[
            _ScaleTemperature('scale'),
            _AddToTemperature('offset', self.modal_volume),
        ],
    )
    self._advance_from_rest(model)

    # Scaling is applied before the offset is added, and both fixers observe
    # diagnostics computed from tendencies of the same step.
    expected = (3.0 * np.array(HEATING_RATES_SI) + offset_rate) * 3600
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)

  @parameterized.named_parameters(
      dict(testcase_name='si_units', sim_units=units.SI_UNITS),
      dict(testcase_name='default_units', sim_units=units.DEFAULT_UNITS),
  )
  def test_energy_balance_fixer_converts_imbalance_to_heating(self, sim_units):
    ylm_map = self._ylm_map(sim_units)
    # Uniform nondimensional column energy imbalance. Surface pressure of the
    # resting state is exp(0) = 1 in nondimensional units.
    imbalance = 1e-3
    model = self._make_model(
        _zero_tendencies,
        sim_units=sim_units,
        tendency_diagnostics={
            'imbalance': diagnostics.InstantDiagnostic(
                extract=_ExtractConstant(
                    'imbalance',
                    cx.field(jnp.full(self.grid.shape, imbalance), self.grid),
                ),
                extract_coords={'imbalance': self.grid},
            ),
        },
        tendency_fixers=[
            fixers.TemperatureAdjustmentForEnergyBalance(
                ylm_map=ylm_map, levels=self.levels, sim_units=sim_units
            )
        ],
    )
    self._advance_from_rest(model)

    dt = sim_units.nondimensionalize_timedelta64(ONE_HOUR)
    heating = imbalance * sim_units.gravity_acceleration / sim_units.Cp
    expected = np.full(len(HEATING_RATES_SI), heating * dt)
    self._assert_uniform_on_levels(self._observe_temperature(model), expected)

  def test_state_fixers_modify_state_after_dycore_step(self):
    model = self._make_model(
        self._heating(),
        parameterization_timestep=QUARTER_HOUR,
        step_start_diagnostics={
            'start': diagnostics.InstantDiagnostic(
                extract=_ExtractTemperature('start', from_prognostics=False),
                extract_coords={'start': self.modal_volume},
            ),
        },
        state_fixers=[_SetTemperature('start', self.modal_volume)],
    )
    self._advance_from_rest(model)

    # Restoring temperature from the start of each step undoes all heating.
    np.testing.assert_allclose(self._observe_temperature(model), 0.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='no_remat', remat_kwargs={}),
      dict(
          testcase_name='remat_substep',
          remat_kwargs=dict(remat_substep=True),
      ),
      dict(
          testcase_name='remat_to_aggregate',
          remat_kwargs=dict(remat_to_aggregate=True),
      ),
      dict(
          testcase_name='remat_advance',
          remat_kwargs=dict(remat_advance=True),
      ),
  )
  def test_gradient_with_respect_to_physics_parameters(self, remat_kwargs):
    model = self._make_model(
        self._heating(),
        aggregate_every=HALF_HOUR,
        parameterization_timestep=QUARTER_HOUR,
        **remat_kwargs,
    )
    inputs = {NONDIM_PROGNOSTICS_KEY: self._resting_state()}
    level_weights = jnp.arange(1.0, 5.0)

    def loss(model):
      model.assimilate(inputs)
      model.advance()
      temperature = model.model_ylm_map.to_nodal(
          model.prognostic_values()['temperature']
      )
      return jnp.sum(level_weights * jnp.mean(temperature.data, axis=(1, 2)))

    grads = nnx.grad(loss)(model)

    # Temperature on each level is `rate * 3600`, so the gradient with respect
    # to the heating rate on level `i` is `level_weights[i] * 3600`.
    np.testing.assert_allclose(
        grads['learned_physics']['rates'][...], level_weights * 3600, rtol=1e-4
    )

  def test_raises_on_invalid_learned_physics_outputs(self):
    def missing_temperature(inputs):
      tendencies = _zero_tendencies(inputs)
      del tendencies['temperature']
      return tendencies

    model = self._make_model(missing_temperature)
    model.assimilate({NONDIM_PROGNOSTICS_KEY: self._resting_state()})
    with self.assertRaisesRegex(
        ValueError, 'must return tendencies for all prognostic variables'
    ):
      model.advance()

    def nodal_temperature(inputs):
      tendencies = _zero_tendencies(inputs)
      tendencies['temperature'] = cx.field(
          jnp.zeros(self.nodal_volume.shape), self.nodal_volume
      )
      return tendencies

    model = self._make_model(nodal_temperature)
    model.assimilate({NONDIM_PROGNOSTICS_KEY: self._resting_state()})
    with self.assertRaisesRegex(
        ValueError, "`learned_physics` tendency of 'temperature' has coordinate"
    ):
      model.advance()

  def test_raises_on_incompatible_timesteps(self):
    with self.assertRaisesRegex(ValueError, 'must be a multiple of'):
      self._make_model(
          _zero_tendencies, aggregate_every=np.timedelta64(40, 'm')
      )
    with self.assertRaisesRegex(ValueError, 'must divide aggregate_every'):
      self._make_model(
          _zero_tendencies,
          aggregate_every=HALF_HOUR,
          parameterization_timestep=np.timedelta64(20, 'm'),
      )


if __name__ == '__main__':
  absltest.main()
