.. _Common Pitfalls:

Common Pitfalls
===============

Here are some common issues and possible solutions for them.

- When debugging, you should use the smallest mesh size possible to minimize time invested.

- Start with a simple optimization case and gradually add design variables and constraints while examining the change in optimized result. The design trade-offs in a simpler optimization problem should be easier to understand.

- If you are running optimizations, it may be less computationally expensive to use `force_fd = True` within the problem dictionary.

- Use `plot_all.py` and `OptView.py` often to visualize your results. Make sure your model is set up like you expect.

- After running an optimization case, always check to see if your constraints are satisfied. pyOptSparse prints out this information automatically, but if you're using Scipy's optimizer, you may need to manually print the constraints.

- Check out http://openmdao.readthedocs.io/en/latest/usr-guide/tutorials/recording.html#the-casereader for more info about how to access the saved data in the outputted `.db` file.
