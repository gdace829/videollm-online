import os, torchvision, transformers
torchvision.set_video_backend('video_reader')
from functools import partial
import gradio as gr

from data.utils import ffmpeg_once

from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.app --resume_from_checkpoint ... 
# 初始化live推理器
liveinfer = LiveInfer()
# 网页中某些组件的样式    
# #gr_title：标题居中显示；
  #gr_video：限制视频框最大高度为 480px；
  #gr_chatbot：限制聊天框最大高度为 480px。
# 

css = """
    #gr_title {text-align: center;} 
    #gr_video {max-height: 480px;}
    #gr_chatbot {max-height: 480px;}
"""
# 获取视频当前时间
get_gr_video_current_time = """async (video, _) => {
  const videoEl = document.querySelector("#gr_video video");
  return [video, videoEl.currentTime];
}"""
# 创建 Gradio Blocks 布局 使用 gr.Blocks() 构建一个完整的 Web 页面布局； 
# 设置页面标题为 "VideoLLM-online"；
# 应用上面定义的 CSS
with gr.Blocks(title="VideoLLM-online", css=css) as demo:
    # 页面顶部标题 Markdown
    gr.Markdown("# VideoLLM-online: Online Video Large Language Model for Streaming Video", elem_id='gr_title')
    with gr.Row():
        with gr.Column():
            # 左侧列：视频展示 + 示例 + 提示信息
            # 用户可以上传视频；
            gr_video = gr.Video(label="video stream", elem_id="gr_video", visible=True, sources=['upload'], autoplay=True)
            # 示例视频按钮 提供几个预设的视频示例，点击即可加载到视频组件中
            # 自动触发后续的处理流程。
            gr_examples = gr.Examples(
                examples=[["demo/assets/cooking.mp4"], ["demo/assets/bicycle.mp4"], ["demo/assets/egoexo4d.mp4"]],
                inputs=gr_video,
                outputs=gr_video,
                label="Examples"
            )
            # 使用提示说明
            gr.Markdown("## Tips:")
            gr.Markdown("- When you upload/click a video, the model starts processing the video stream. You can input a query before or after that, at any point during the video as you like.")
            gr.Markdown("- **Gradio refreshes the chatbot box to update the answer, which will delay the program. If you want to enjoy faster demo as we show in teaser video, please use https://github.com/showlab/videollm-online/blob/main/demo/cli.py.**")
            gr.Markdown("- This work is primarily done at a university, and our resources are limited. Our model is trained with limited data, so it may not solve very complicated questions. However, we have seen the potential of 'learning in streaming'. We are working on new data method to scale streaming dialogue data to our next model.")
        # 右侧列：聊天界面 + 参数调节器
        with gr.Column():
            # 聊天机器人组件 
            # fn=liveinfer.input_query_stream：指定当用户输入问题时调用的函数；
            # chatbot：自定义聊天框外观（头像、最大高度等）；
            # examples：提供一些示例问题让用户快速尝试。
            gr_chat_interface = gr.ChatInterface(
                fn=liveinfer.input_query_stream,
                chatbot=gr.Chatbot(
                    elem_id="gr_chatbot",
                    label='chatbot',
                    avatar_images=('demo/user_avatar.png', 'demo/assistant_avatar.png'),
                    render=False
                ),
                examples=['Please narrate the video in real time.', 'Please describe what I am doing.', 'Could you summarize what have been done?', 'Hi, guide me the next step.'],
            )
            
            def gr_frame_token_interval_threshold_change(frame_token_interval_threshold):
                liveinfer.frame_token_interval_threshold = frame_token_interval_threshold
            # 流媒体阈值参数调节器 控制视频帧采样频率的阈值；当用户拖动滑块时，会调用下面的函数更新配置：
            gr_frame_token_interval_threshold = gr.Slider(minimum=0, maximum=1, step=0.05, value=liveinfer.frame_token_interval_threshold, interactive=True, label="Streaming Threshold")
            gr_frame_token_interval_threshold.change(gr_frame_token_interval_threshold_change, inputs=[gr_frame_token_interval_threshold])
        # 隐藏变量：用于控制视频播放时间 & 刷新机制
        gr_video_time = gr.Number(value=0, visible=False)
        gr_liveinfer_queue_refresher = gr.Number(value=False, visible=False)
        # 当用户上传或选择了一个新视频时触发的事件处理函数
        # src_video_path: 用户上传的原始视频路径； history: 当前的问答历史记录； 
        # video_time: 视频当前播放的时间点；gate: 控制刷新机制的开关变量（避免死循环）
        def gr_video_change(src_video_path, history, video_time, gate):
            # 使用 os.path.splitext() 将文件名与扩展名分离。
            name, ext = os.path.splitext(src_video_path)
            # 构造一个目标路径，用于保存转码后的视频。
            # 路径中包含帧率（FPS）和分辨率信息，确保不同设置下的视频不会冲突。
            ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
            # 转码原始视频 设置分辨率帧率 cooking_2fps_384.mp4 如果存在视频则不需要转码
            # 示例视频是存在的 
            if not os.path.exists(ffmpeg_video_path):
                # 创建目标目录（如果不存在）；exist_ok=True 表示即使目录已存在也不会报错。
                os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                # 调用 ffmpeg_once() 函数，使用 FFmpeg 对视频进行转码； 设置目标帧率（FPS）和分辨率。生成转码后的地址 还是mp4
                ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
                # 输出一条日志信息，显示视频转换完成，并打印帧率和分辨率。
                logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
            # 将/cooking.mp4 变为 /cooking_2fps_384.mp4  2  384
            logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolutionsjs')
            # 调用 liveinfer.load_video()，加载转码后的视频，准备进行推理。 将视频处理成tensor
            liveinfer.load_video(ffmpeg_video_path) # live是同一个liveinfer，之后继续上传时需要等待上一个视频处理完毕。
            # 向模型输入视频流，参数 0 可能表示从第 0 帧开始。
            liveinfer.input_video_stream(0)
            # 使用live进行推理 推理的时候出现问题
            query, response = liveinfer()
            if query or response:
                history.append((query, response))# 如果有新的问答内容，就追加到聊天记录 history 中。
            # 返回值是一个 (query, response) 元组，表示模型自动产生的问题和回答（或者可能是系统提示词+模型回复）。
            return history, video_time + 1 / liveinfer.frame_fps, not gate
        # 返回三个值作为输出：
        # 更新后的聊天记录 history；
        # 更新后的视频时间戳（增加一帧的时间）；
        # 1 / liveinfer.frame_fps 是每帧的时间间隔（秒）；
        # 切换 gate 的布尔值（True ↔ False），用于触发下一次处理。
        gr_video.change(
            gr_video_change, inputs=[gr_video, gr_chat_interface.chatbot, gr_video_time, gr_liveinfer_queue_refresher], 
            outputs=[gr_chat_interface.chatbot, gr_video_time, gr_liveinfer_queue_refresher]
        )
        
        def gr_video_time_change(_, video_time):
            liveinfer.input_video_stream(video_time)
            return video_time
        gr_video_time.change(gr_video_time_change, [gr_video, gr_video_time], [gr_video_time], js=get_gr_video_current_time)

        def gr_liveinfer_queue_refresher_change(history):
            while True:
                query, response = liveinfer()
                if query or response:
                    history[-1][1] += f'\n{response}'
                yield history
        gr_liveinfer_queue_refresher.change(gr_liveinfer_queue_refresher_change, inputs=[gr_chat_interface.chatbot], outputs=[gr_chat_interface.chatbot])
    
    demo.queue()
    demo.launch(share=True)
