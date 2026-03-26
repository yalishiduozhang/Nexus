# MMEB v2 全量评测媒体下载计划

更新时间：2026-03-26

## 目标

将 MMEB v2 评测所需的官方媒体文件下载到大容量磁盘目录：

- `/storage/zhangx_data/MMEB-V2`

说明：

- 这里下载的是 `TIGER-Lab/MMEB-V2` 中的图片、视频帧、视觉文档页面等媒体文件。
- 这些文件是跑 MMEB v2 评测时最占空间、也最关键的一部分。
- 评测时还会用到来自其他 Hugging Face repo 的小体量元数据，但它们远小于媒体数据，后续可以继续本地镜像化。

## 预计体量

按 Hugging Face 文件树统计，当前计划下载如下 11 个大包：

- `image-tasks/mmeb_v1.tar.gz`
- `video-tasks/frames/video_cls.tar.gz`
- `video-tasks/frames/video_ret.tar.gz`
- `video-tasks/frames/video_mret.tar.gz`
- `video-tasks/frames/video_qa.tar.gz-00`
- `video-tasks/frames/video_qa.tar.gz-01`
- `video-tasks/frames/video_qa.tar.gz-02`
- `video-tasks/frames/video_qa.tar.gz-03`
- `video-tasks/frames/video_qa.tar.gz-04`
- `visdoc-tasks/visdoc-tasks.data.tar.gz`
- `visdoc-tasks/visdoc-tasks.images.tar.gz`

体量估算：

- image 约 `6.55 GiB`
- video 约 `85.33 GiB`
- visdoc 约 `33.10 GiB`
- 合计压缩包约 `124.98 GiB`

备注：

- Hugging Face 文件树里还有 `visdoc-tasks.tar.gz-00/01/02` 三个分片文件。
- 从总字节数判断，它们大概率与 `visdoc-tasks.images.tar.gz` 对应，不应重复下载计入。
- 因此当前计划先按“不重复下载”的保守方案执行。

## 存储检查

当前磁盘空间：

- `/storage` 可用空间约 `8.4T`
- `/home` 可用空间约 `76G`

因此本次下载必须落到 `/storage`，不能继续放在 `/home`。

## 下载脚本

仓库内新增脚本：

- `tools/multimodal_retrieval/download_mmeb_v2_eval_media.sh`

脚本作用：

- 逐个下载 MMEB v2 官方评测媒体大包
- 自动创建目录
- 优先使用 `aria2c`，不可用时回退到 `curl`
- 支持断点续传
- 将日志记录到目标目录下的 `logs/download.log`

## 当前执行状态

- 已完成目标路径与空间检查
- 已确认下载来源与文件列表
- 已启动后台下载任务
- `tmux` 会话名：`mmeb_v2_download`
- 下载目录：`/storage/zhangx_data/MMEB-V2`
- 日志文件：`/storage/zhangx_data/MMEB-V2/logs/download.log`

当前已观察到：

- 第一包 `image-tasks/mmeb_v1.tar.gz` 已开始下载
- 目标目录已创建成功

## 进度查看方式

可用如下命令查看进度：

- `tmux capture-pane -pt mmeb_v2_download`
- `tail -n 50 /storage/zhangx_data/MMEB-V2/logs/download.log`
- `du -sh /storage/zhangx_data/MMEB-V2`

说明：

- 下载脚本支持断点续传。
- 即使网络短暂中断，重新进入同一命令继续执行即可。

## 后续计划

1. 等全部媒体文件下载完成
2. 校验文件是否齐全
3. 如有需要，再补充解压脚本与目录整理脚本
4. 再继续推进“全量 metadata 本地化”与“全量 MMEB v2 eval 闭环”
