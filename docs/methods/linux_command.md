# Linux 操作记录

## 环境信息查询

### 查看内核/操作系统/CPU信息

```shell
uname -a
```

### 查看操作系统版本

```shell
head -n 1 /etc/issue
```

### 查看 CPU 信息

```shell
cat /proc/cpuinfo
```

## 杂项

### 查找端口占用

```shell
lsof -i:端口号
```

### 删除显卡上所有的进程

```shell
fuser -v /dev/nvidiaX | xargs -t -n 1 kill -9
```

其中大写的 `X` 替换为显卡编号（`0-7`）

### 删除显卡上某个用户所有的进程

```shell
fuser -v /dev/nvidiaX 2>&1 | grep zhr | xargs -t -n 1 kill -9
```

貌似存在一些问题，主要是 `xargs` 没有正确 parse 出 PID，把所有内容都放进去 Kill 了一遍。

