import torch, torchvision, transformers, collections
torchvision.set_video_backend('video_reader')
from dataclasses import asdict
from torchvision.io import read_video

from models import build_model_and_tokenizer, parse_args, fast_greedy_generate

logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

class LiveInfer:
    def __init__(self, ) -> None:
        args = parse_args()
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        self.model.connector.to_empty(device="cpu")
        self.model.connector.apply(self.model._init_weights)
        self.model.to('cuda')
        
        # visual
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(self.model.config.frame_num_tokens).reshape(1,-1)
        
        # generation
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device='cuda', dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')
        
        # app
        self.reset()
    # 根据用户的提问（query）或模型上文，调用大语言模型进行推理，生成对应的回复（response），并附带时间戳信息。
    # video_time: 视频当前的时间点（秒），用于标注输出
    # query: 用户的问题（字符串），可能为 None，表示不是用户主动提问，而是模型自动生成的内容。
    def _call_for_response(self, video_time, query):

        # 如果有用户提问
        if query is not None:
            # 如果用户提供了问题，则使用 tokenizer 将其转换成 token ID； 
            # 使用 apply_chat_template 按照聊天模板格式化输入；
            # 添加特殊提示词（如 <|assistant|>）以引导模型生成回复；
            # 最终将 token IDs 转移到 GPU 上供模型使用。
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
        else:# ：如果 query 是 None
            # 这是一个 hack 写法，表示当没有用户提问时，假设上一轮输出的 token 是 933（可能是某个特殊标记，比如 ]\n）；
            # 然后将它替换为一段预定义的 prompt（_added_stream_generation_ids），用来引导模型继续生成描述性内容
            # 期望 self.last_ids 的值是 933； 实际上它的值是 tensor(, device='cuda:0')；
            assert self.last_ids == 933, f'{self.last_ids} != 933' # HACK, 933 = ]\n
            self.last_ids = self._added_stream_generation_ids
        print("sjs 生成回复")
        # 构建输入 embeddings 并调用模型生成回复 获取 token ID 对应的 embedding 向量。
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        # 推理生成回复 
        # 使用 fast_greedy_generate() 快速生成回复； 
        # 传入当前输入、历史缓存、结束符等参数； 
        # output_ids: 生成的 token 序列； past_key_values: 更新后的 attention cache，供下一次推理使用。
        output_ids, self.past_key_values = fast_greedy_generate(model=self.model, inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids)
        # 记录最新输出 token ,只保留最后一个生成的 token ID，作为下一轮推理的输入。
        self.last_ids = output_ids[:, -1:] 

        #  构造输出格式（带时间戳） 如果有提问，则在 query 前加上时间戳和角色标识； 解码模型输出，得到自然语言回复，并加上时间戳和角色标识。
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)}'
        # 返回 (query, response)，供前端或其他模块展示或记录。
        return query, response
    
    def _call_for_streaming(self, ):
        # self.frame_embeds_queue: 一个队列，存储着视频帧的时间戳和对应的嵌入向量。
        # self.query_queue: 用户提问的队列，也按时间排序。
        # self.model: 大语言模型（LLM），用于处理输入并生成回复。
        # self.past_key_values: 缓存注意力机制中的 key/value，用于加速连续推理。
        # self.last_ids: 上一次模型输出的 token ID。
        # self.frame_token_interval_id: 一个特殊的 token ID，表示“继续等待下一帧”。
        # self.frame_token_interval_threshold: 控制是否跳过当前帧的阈值。
        # 每个元素格式：(video_time, frame_embeds)

        # 主循环：只要有帧就持续处理
        while self.frame_embeds_queue:

            # 如果有提问早于下一个视频帧 → 优先返回该提问
            # 判断是否有用户提问；
            # 如果最早的问题发生在下一个视频帧之前，则立即返回该问题，让模型优先回答它。
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            
            #  取出当前视频帧及其时间戳 从队列中取出一帧数据：
            #  video_time: 当前帧的时间点（秒）；
            #  frame_embeds: 帧的嵌入向量（来自视觉编码器）。
            video_time, frame_embeds = self.frame_embeds_queue.popleft()

            # 初始化模型输入（第一次或结束符后）
            # 如果是第一帧，使用初始 token（如 <BOS>）开始生成；
            # 如果上一轮生成以 EOS 结束，添加一段提示词（prompt）引导模型继续生成。
            if not self.past_key_values:
                self.last_ids = self._start_ids
            elif self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)

            # 构建输入 embeddings
            # 将上一轮输出的 token 转换为 embedding；
            # 和当前帧的 embedding 拼接在一起作为新输入；
            # 输入到 LLM 中进行推理。
            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
                frame_embeds.view(1, -1, self.hidden_size),
            ], dim=1)

            # 执行一次推理，保存 attention cache 
            # 用大语言模型进行推理； 
            # 保留 attention cache，用于下一轮推理（提升效率）。
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values

            # 再次检查是否有提问正好发生在此帧之后 
            # 如果当前帧的时间戳大于等于最早提问的时间，则立即返回该提问；
            # 表示：“虽然我在描述视频，但你问的问题正好发生在这帧之后，我先处理你的问题”。
            # 2. if the same time, response after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            # 获取下一个 token 的概率分布 
            # 获取最后一个 token 的 logits 并 softmax 得到概率分布。
            # 3. if the next is frame but next is not interval, then response
            next_score = outputs.logits[:,-1:].softmax(dim=-1)
            # 屏蔽低置信度的帧间隔 token
            if next_score[:,:,self.frame_token_interval_id] < self.frame_token_interval_threshold:
                next_score[:,:,self.frame_token_interval_id].zero_()
            # 获取最终预测的 token ID 选择最大概率的 token 作为当前输出。
            self.last_ids = next_score.argmax(dim=-1)

            # 如果不是“帧间隔” token，说明这是一个有意义的输出（比如描述语句的一部分），则返回；
            # 否则继续循环，处理下一帧。
            if self.last_ids != self.frame_token_interval_id: 
                return video_time, None
        return None, None
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            frames_embeds = self.model.visual_embed(self.video_tensor[ranger]).split(self.frame_num_tokens)
            self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)])
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    # 加载视频 
    # /tmp/gradio/19605b82d1d6a550eaba8a8698ae360159f47d8782df59d3f7c7f97479c289d6/cooking_2fps_384.mp4 ->
    #  video_tensor 的shape 为 torch.Size([215, 3, 384, 384]), 2 FPS

    # 将一个视频文件加载到内存中，并将其转换为模型可以处理的张量格式（Tensor），同时记录一些基本信息。
    def load_video(self, video_path):
        # output_format='TCHW'： 张量格式（Tensor）
        # 表示输出格式为：
        # T: Time（帧数）
        # C: Channel（通道数，如 RGB 是 3）
        # H: Height（高度）
        # W: Width（宽度）
        # 视频有 150 帧，分辨率为 224x224，RGB 三通道，则 tensor 形状为：(150, 3, 224, 224)
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0].to('cuda')
        self.num_video_frames = self.video_tensor.size(0)# 总帧数 num_video_frames
        self.video_duration = self.video_tensor.size(0) / self.frame_fps# 视频时长 video_duration；
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')
    #
    # Python 类中的魔法方法（magic method），
    # 允许一个类的实例像函数一样被调用liveinfer = LiveInfer(...)
    # query, response = liveinfer()  # 实际上调用的是 __call__      
    def __call__(self, ):# 这块出错
        # 等待视频帧数据流入，在合适时机调用 _call_for_streaming 获取问题或时间戳，再调用 _call_for_response 生成模型回复。
        while not self.frame_embeds_queue:
            # 这是一个阻塞循环：只要 frame_embeds_queue 是空的（即还没有视频帧进来），就一直等待。
            # 一旦有视频帧被放入队列，就会跳出循环，继续执行下面的逻辑。
            # 确保有视频帧可以处理，避免空指针错误。
            continue
        # 调用 _call_for_streaming() 获取当前时间和提问 它会尝试从视频帧队列中取出一帧，并进行推理；
        # 可能返回：
        # (video_time, None)：表示模型生成了一段描述；
        # (video_time, query)：表示用户在某个时间点提出了问题。
        video_time, query = self._call_for_streaming()
        response = None

        #  如果获取到了时间戳，则进一步调用 _call_for_response()
        # 如果 video_time 不为 None，说明当前帧有效；
        # 如果 query 不为 None，说明用户提出了问题，需要生成回答；
        # 如果 query 为 None，说明是模型自动生成的内容，也要继续推理（见 _call_for_response 中的 hack 处理）；
        if video_time is not None:# 这块出错
            query, response = self._call_for_response(video_time, query)
        # 最终返回 (query, response)： 
        # 如果用户提问了：返回 (提问内容, 模型回答) 
        # 如果没有提问：返回 (None, 模型生成的视频描述)
        return query, response