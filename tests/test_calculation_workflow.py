import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from calculation_workflow import build_input_script


class SequentialWorkflowTest(unittest.TestCase):
    def fake_scf(self, settings):
        dft = SimpleNamespace(converged=True, niter=2, Kinetic_energy=1.0,
            Nuc_energy=-3.0, J_energy=0.5, XC_energy=-0.2,
            Nuclear_repulsion_energy=0.4, mo_occupations=np.array([2., 0.]),
            mo_energies=np.array([-0.5, 0.1]), scf_energies=[-1.0, -1.3],
            xc=settings['xc_functional'], exx_coef=0.25 if settings['xc_functional'] in ('PBE0', 'B3LYP') else 0., isDF=True)
        return dict(settings=dict(settings), dft=dft, dmat=np.eye(2), energy=-1.3,
            elapsed=0.01, actions=[], cubes={}, forces=None, dipole=None,
            comparison=None, log='SCF done')

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def test_actions_reuse_scf_and_saved_settings(self):
        with patch('calculation_workflow.run_scf', side_effect=self.fake_scf) as scf, \
             patch('calculation_workflow.calculate_dipole', return_value=np.array([0., 0., 1.])) as dipole, \
             patch('calculation_workflow.calculate_forces', return_value=np.array([[0.01, 0., 0.], [-0.02, 0., 0.], [0., 0.01, 0.]])) as forces, \
             patch('calculation_workflow.generate_cube', return_value='cube') as cube:
            app = AppTest.from_file('Home.py').run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(any(b.label == 'Calculate forces' for b in app.button))
            self.assertEqual(next(w for w in app.number_input if w.label == 'Max Iterations:').value, 20)
            self.assertEqual(next(w for w in app.selectbox if w.label == 'AO Basis Type:').value, 'SAO')
            self.button(app, '🚀 Run DFT Calculation').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(scf.call_count, 1)
            cube.assert_not_called()
            dipole.assert_not_called()
            initial = build_input_script(app.session_state.scf_result)
            self.assertNotIn('Utils', initial)
            self.assertNotIn('DFT_Grad', initial)
            self.assertNotIn('DF_algo', initial)
            next(w for w in app.selectbox if w.label == 'XC Functional:').select('PBE0').run()
            self.button(app, 'Calculate dipole moment').click().run()
            self.assertFalse(app.exception)
            self.button(app, 'Calculate forces').click().run()
            self.assertFalse(app.exception)
            next(w for w in app.slider if w.label == 'Force arrow scale').set_value(1.5).run()
            self.assertFalse(app.exception)
            self.assertEqual(scf.call_count, 1)
            self.assertEqual(dipole.call_count, 1)
            self.assertEqual(forces.call_count, 1)
            self.assertEqual(app.session_state.scf_result['settings']['xc_functional'], 'LDA')
            # Avoid parsing mocked cube content in the browser visualization helper.
            with patch('py3Dmol.view') as viewer:
                viewer.return_value._make_html.return_value = '<div></div>'
                self.button(app, 'Plot electron density').click().run()
                self.assertFalse(app.exception)
                next(w for w in app.slider if w.label == 'Opacity:').set_value(0.5).run()
                self.assertFalse(app.exception)
            self.assertEqual(cube.call_count, 1)
            self.assertEqual(scf.call_count, 1)
            final = build_input_script(app.session_state.scf_result)
            compile(final, '<download>', 'exec')
            self.assertIn('DFT_Grad(dft)', final)
            self.assertIn('get_dipole_moment', final)
            self.assertIn('write_density_cube', final)
            self.button(app, '🚀 Run DFT Calculation').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(scf.call_count, 2)
            self.assertEqual(app.session_state.scf_result['actions'], [])
            self.assertEqual(app.session_state.scf_result['cubes'], {})
            self.assertTrue(self.button(app, 'Calculate forces').disabled)

    def test_failed_action_can_retry_without_losing_scf(self):
        with patch('calculation_workflow.run_scf', side_effect=self.fake_scf) as scf, \
             patch('calculation_workflow.calculate_dipole', side_effect=[RuntimeError('temporary failure'), np.zeros(3)]) as dipole:
            app = AppTest.from_file('Home.py').run(timeout=30)
            self.button(app, '🚀 Run DFT Calculation').click().run()
            self.button(app, 'Calculate dipole moment').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('temporary failure' in e.value for e in app.error))
            self.assertEqual(app.session_state.scf_result['actions'], [])
            self.button(app, 'Calculate dipole moment').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(dipole.call_count, 2)
            self.assertEqual(scf.call_count, 1)
            self.assertEqual(app.session_state.scf_result['actions'], [{'kind': 'dipole'}])

    def test_scripts_cover_settings_and_requested_actions(self):
        for functional in ['LDA', 'HF', 'B3LYP', 'PBE0']:
            for guess in ['sano', 'core']:
                for grid in [0, 3, 5]:
                    result = self.fake_scf(dict(xyz_content='2\nH2\nH 0 0 0\nH 0 0 0.74\n',
                        basis_set='sto-3g', auxbasis='def2-universal-jfit', xc_functional=functional,
                        grid_level=grid, initial_guess=guess, conv_crit=1e-6,
                        max_iterations=20, ncores=1, use_sao_basis=True))
                    for action in [None, {'kind': 'dipole'}, {'kind': 'forces'},
                                   {'kind': 'cube', 'orbital': 0, 'resolution': 30},
                                   {'kind': 'cube', 'orbital': None, 'resolution': 40},
                                   {'kind': 'comparison', 'xc': functional}]:
                        result['actions'] = [] if action is None else [action]
                        compile(build_input_script(result), '<download>', 'exec')


if __name__ == '__main__':
    unittest.main()
