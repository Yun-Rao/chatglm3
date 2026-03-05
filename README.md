环境安装
```bash
pip install -r requirements.txt
```
克隆 LLaMA-Factory
在终端进入项目目录运行代码
```bash
git clone --branch v0.8.3 https://github.com/hiyouga/LLaMA-Factory.git
```
进入LLaMA-Factory目录运行代码
```python
pip install -e ".[torch,metrics]"
```
下载模型
```bash
python download-model.py
```
#微调

```bash
llamafactory-cli train ../configs/chatglm_lora_sft.yaml
```

MySQL 初始化
```bash
apt-get install -y mysql-server
service mysql start
mysql -u root -e "CREATE DATABASE rag_system CHARACTER SET utf8mb4;"
mysql -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'your_password';"

# 设置管理员账号
mysql -u root -p rag_system -e "UPDATE users SET is_admin=1 WHERE username='your_username';"

# 给 messages 表加 cot 字段（升级时执行一次）
mysql -u root -p rag_system -e "ALTER TABLE messages ADD COLUMN cot TEXT NULL;"

cd /root/autodl-tmp/ChatGLM/rag_system/backend
python main.py

# 设置管理员（注册账号后执行）
mysql -u root -p rag_system -e "UPDATE users SET is_admin=1 WHERE username='your_username';"
```