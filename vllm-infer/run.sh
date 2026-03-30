

# 基本使用（交互式确认）
bash kill_multi_nodes.sh

# 跳过确认直接执行
bash kill_multi_nodes.sh -y

# 只终止 ray 相关进程
bash kill_multi_nodes.sh -y -k "ray"

# 自定义节点列表文件
bash kill_multi_nodes.sh /path/to/nodes.txt