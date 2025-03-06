# Linux 操作记录

## 环境信息查询

### 查看内核/操作系统/CPU信息

```console
uname -a
```

### 查看操作系统版本

```console
head -n 1 /etc/issue
```

### 查看 CPU 信息

```console
cat /proc/cpuinfo
```

## CUDA 相关

### 删除显卡上所有的进程

```console
fuser -v /dev/nvidiaX | xargs -t -n 1 kill -9
```

其中大写的 `X` 替换为显卡编号（`0-7`）

### 删除显卡上某个用户所有的进程

```bash
fuser -v /dev/nvidiaX 2>&1 | grep zhr | xargs -t -n 1 kill -9
```

貌似存在一些问题，主要是 `xargs` 没有正确 parse 出 PID，把所有内容都放进去 Kill 了一遍。

### 禁用损坏的卡

可以先使用下面的命令查看系统日志，找出报错显卡的 ID，可以重点观察 `rm_init_adapter failed` 等字样。

```bash
dmest -T
```

找到的 ID 的格式类似 `0000:0b:00.0`，随后通过下面的命令禁用坏卡：

```bash
sudo nvidia-smi drain -p 0000:0b:00.0 -m 1
```

## 杂项

### 查找端口占用

```console
lsof -i:端口号
```

### 挂载硬盘

首先通过 `fdisk -l` 查看已有硬盘信息，找到要挂在的硬盘，例如 `/dev/sdb1`

然后通过 `mount` 命令进行挂载，例如将硬盘 `dev/sdb1` 挂载到 `/home` 目录下的命令可以写为：

```console
mount /dev/sdb1 /home
```
