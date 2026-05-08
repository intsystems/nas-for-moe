# RSM серверы

Кластер `rsm3`, `rsm4`, `rsm5`, `rsm6` — альтернативная площадка для экспериментов
(помимо Jarvis, см. @server.md).

## Подключение

```bash
ssh rsm6
```

Хосты прописаны в `~/.ssh/config`:

| Alias | HostName | User | Key |
|---|---|---|---|
| rsm3 | 192.168.234.3 | pbabkin | `~/.ssh/id_rsa` |
| rsm4 | 192.168.234.4 | pbabkin | `~/.ssh/id_rsa` |
| rsm5 | 192.168.234.5 | pbabkin | `~/.ssh/id_rsa` |
| rsm6 | 192.168.234.6 | pbabkin | `~/.ssh/id_rsa` |

Пароль (для `rsm3`, хранится в `~/.ssh/config` комментарием): `Agro_Digit@ls`.

## Использование

Команды на удалённой машине — через `ssh rsm6 "<command>"` или интерактивную сессию.
Если нужно запустить длительный процесс — используй `nohup` / `tmux` / `screen`,
чтобы он не завис при разрыве ssh.

Для копирования файлов — `scp` / `rsync`:

```bash
rsync -avz code/ rsm6:~/nas-for-moe/code/
```

## Specs

> TBD: заполнить после первого успешного подключения (hostname, CPU, RAM, GPU, CUDA).
