# 更新所有可以更新的Python库
import os
from pip._internal.utils.misc import get_installed_distributions

out_list = get_installed_distributions()

for out_dist in out_list:
    project = out_dist.project_name
    print(project + '\n')
    action = os.popen('python -m pip install --upgrade ' + project)
    result = action.read()
    print(result)
    