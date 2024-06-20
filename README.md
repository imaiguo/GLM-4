# GLM-4

## 1 环境准备

```bash
> cd /opt/Data/PythonVenv
> python3 -m venv glm4
> source /opt/Data/PythonVenv/glm4/bin/activate
>
```

## 2 部署推理环境

```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
>
```

## 3 启动服务

```bash
> python basic_demo/openai_api_server.py
> python basic_demo/openai_api_server_transformer.py
> python basic_demo/WebGradio.py
> jupyter notebook --no-browser --port 7000 --ip=192.168.2.199
```

## 4 命令交互

```bash
> python basic_demo/trans_cli_demo.py
```