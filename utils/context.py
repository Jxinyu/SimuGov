import contextvars

# 定义一个上下文变量，默认值为 None
# 这个变量对于每个异步任务链是独立的
current_sim_subdir = contextvars.ContextVar("current_sim_subdir", default=None)