import contextvars

# Define a context variable, default value is None
# This variable is independent for each asynchronous task chain
current_sim_subdir = contextvars.ContextVar("current_sim_subdir", default=None)