# Import the Scipy Optimizer and set the driver of the problem to use
# it, which defaults to an SLSQP optimization method
from openmdao.api import ScipyOptimizeDriver
prob.driver = ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9

recorder = SqliteRecorder("aero_analysis_test.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
prob.model.add_constraint(point_name + '.wing_perf.CL', equals=0.5)
prob.model.add_objective(point_name + '.wing_perf.CD', scaler=1e4)
