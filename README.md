# 日语字幕离线翻译器

这是一个本地离线运行的 Windows/Python 小工具，用 Ollama 模型把日语字幕翻译成简体中文。默认模型是 `qwen2.5:7b`，也可以切换成 `qwen2.5`、`qwen2.5:latest`、`hunyuan-mt`、`qwen25coder14b` 或你本地 Ollama 里的其他模型。

## 支持格式

- `.srt`
- `.vtt`
- `.ass`
- `.ssa`

## 使用前准备

确认 Ollama 服务在运行：

```powershell
ollama serve
```

确认模型存在：

```powershell
ollama list
```

如果没有模型，先拉取：

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5
ollama pull hunyuan-mt
```

## 图形界面

双击：

```text
launch_hidden.vbs
```

这个启动方式不会显示黑色命令行窗口。`启动翻译器.bat` 也可以继续用，但双击 `.bat` 时 Windows 可能会短暂闪一下命令行窗口。

推荐设置：

- 模型名：默认 `qwen2.5:7b`。如果本地模型名是 `qwen2.5` 或 `qwen2.5:latest`，可以在下拉框里直接切换。
- GPU 层数：`999` 表示尽量使用显卡，`0` 表示 CPU。
- 每批字幕数：默认 `30`，建议 `20` 到 `40`。越大请求越少，但上下文越长，也越容易漏编号。
- 单条超时秒数：默认 `600`，遇到 `timed out` 可以改成 `900` 或 `1200`。
- 失败重试次数：默认 `3`。
- 断点续跑：建议保持勾选。
- 使用翻译缓存：建议保持勾选。重复字幕会直接读取缓存，不再调用模型。

## 生成字幕

点击主界面的“生成字幕”按钮，可以从视频/音频生成日语 `.srt` 字幕。

推荐设置：

- Whisper 模型：`medium`
- 语言：`ja`
- 设备：`cuda`
- 精度：`float16`
- 识别模式：`标准`
- 音频增强：开启

成人视频音轨常见背景音、人声距离变化、喘息声和音乐干扰。如果字幕不准，优先试：

- Whisper 模型：`large-v3`
- 识别模式：`高准确率`
- 音频增强：开启

RTX 3070 上不建议一开始就用 `large-v3` 跑长视频。`large-v3` 加载和识别都明显更慢，界面可能长时间停在“正在识别语音”。建议先用 `medium`，如果想更快用 `small`。

如果显存不够或报错，可以改成：

- Whisper 模型：`small`
- 精度：`int8_float16`

生成字幕时软件会显示媒体时长，并在识别过程中按时间轴显示大致百分比。

如果出现：

```text
Library cublas64_12.dll is not found or cannot be loaded
```

说明 faster-whisper 的 CUDA 运行库没有加载成功，通常是 CUDA 12 / cuBLAS / cuDNN 缺失或 PATH 没配好。软件会自动改用 `CPU / int8` 重试生成字幕，速度会慢一些，但不会直接失败。

想继续用 GPU，需要把 CUDA 12 相关运行库安装好，并确认 `cublas64_12.dll` 所在目录能被系统 PATH 找到。

生成完成后，可以把生成的 `.ja.srt` 作为输入字幕，再用翻译功能生成中文字幕。

## 断点续跑

翻译过程中会生成：

```text
你的输出字幕.srt.progress.json
```

每成功翻译一条字幕，程序都会保存一次进度，并更新当前输出字幕。即使中途出现 `timed out`，下次点击“开始翻译”会自动载入进度，从未完成的位置继续，不需要从头跑。

全部完成后，进度文件会自动删除。

## 命令行

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.srt"
```

指定输出文件：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.srt" -o "D:\anime\episode01.zh.srt"
```

使用 GPU、拉长超时、开启重试：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.srt" --num-gpu 999 --timeout 900 --retries 5 --batch-size 30
```

忽略已有进度，从头翻译：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.srt" --no-resume
```

关闭翻译缓存：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.srt" --no-cache
```

从视频生成日语字幕：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.mkv" --transcribe -o "D:\anime\episode01.ja.srt"
```

指定 faster-whisper 模型：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.mkv" --transcribe --whisper-model medium --device cuda --compute-type float16
```

高准确率生成：

```powershell
py -3 jp_subtitle_translator.py "D:\anime\episode01.mkv" --transcribe --whisper-model large-v3 --recognition-mode 高准确率
```

## GPU 加速

程序默认向 Ollama 传入：

```text
num_gpu = 999
keep_alive = 30m
```

确认 GPU 是否真的在工作：

```powershell
nvidia-smi
```

如果显存占用和 GPU 利用率有变化，说明模型正在使用显卡。Ollama 能不能用 GPU 取决于显卡驱动、Ollama 版本、模型格式和显存大小；程序已经把 GPU 参数传给 Ollama，但最终调度由 Ollama 决定。

## 遇到 timed out

这是某一条字幕请求超过等待时间导致的。新版已经做了三层保护：

- 单条失败自动重试。
- 默认超时从 `180` 秒提高到 `600` 秒。
- 每批成功后立刻保存进度，下次自动续跑。

## 批量翻译

新版不再默认一条字幕请求一次，而是按“每批字幕数”合并请求。默认每批 `30` 条。

提示词已改短，要求模型只返回：

```text
编号. 译文
```

程序会按编号解析结果。如果某一批漏掉编号，会自动重试这一批，不会保存残缺结果。

如果一整批连续失败，例如 Ollama 返回 `HTTP Error 502: Bad Gateway`，软件会自动把这一批拆成更小批次继续翻译。比如 40 条失败会拆成 20+20，继续失败再拆成 10、5，直到单条。已成功的小批次会立即保存进度。

## 翻译缓存

软件目录里会自动生成：

```text
translation_cache.json
```

翻译前会先查缓存。相同模型、相同双语设置、相同日文原文以前翻译过，就直接使用缓存译文，不再请求 Ollama。

新翻译成功的字幕会自动写入缓存。这个缓存可以跨不同字幕文件复用，适合字幕里常见的重复短句。

## 翻译用时

翻译完成后，软件日志和完成弹窗会显示本次用时。命令行模式会额外输出 `Elapsed`。

如果同一条字幕一直超时，可以把“单条超时秒数”改成 `900` 或 `1200`，或者把模型换成 `qwen2.5:7b` / `qwen2.5` 试试。
