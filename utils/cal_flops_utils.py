class AttentionFlopsCalculator:
    """
    一个用于根据提供的表格计算不同注意力机制的FLOPs的类。
    H: hidden dimension (隐藏维度)
    N: sequence length (序列长度)
    nq, nk, nv, nc: number of query, key, value, and latent value heads (查询、键、值、潜在值头的数量)
    dh: per-head dimension (每个头的维度)
    dl: latent dimension (潜在维度)
    drope, dnope: MLA 中使用的特定维度
    dc: context dimension (上下文维度)(?)
    """
    def __init__(self):
        self.attention_type = ""
        self.num_hidden_layers = 0
        self.N = 0
        self.H = 0
        self.nq = 0
        self.nk = 0
        self.nv = 0
        self.nc = 0
        self.nh = 0
        self.dh = 0
        self.dl = 0
        self.dc = 0
        self.drope = 0
        self.dnope = 0
        self.kv_cache_per_layer = 0
        self.attention_per_layer = 0
        self.linear_per_layer = 0
        self.total = 0
        self.flops_per_token = 0

    def set_params(self, **kwargs):
        self.attention_type = kwargs["attention_type"]
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 0)
        p = kwargs.get("p_tokens", 0)
        n = kwargs.get("n_tokens", 0)
        self.N = p
        # self.N = p + n
        self.H = kwargs.get("H", kwargs.get("hidden_size", 0))
        self.nq = kwargs.get("nq", kwargs.get("num_attention_heads", 0))
        self.nk = kwargs.get("nk", kwargs.get("num_key_value_heads") / 2)
        self.nv = kwargs.get("nv", kwargs.get("num_key_value_heads") / 2)
        self.nc = kwargs.get("nc", 0)
        self.nh = kwargs.get("nh", kwargs.get("num_attention_heads", 0))
        self.dh = self.H / self.nq if self.nq != 0 else 0
        self.dl = kwargs.get("dl", kwargs.get("hidden_size", 0))
        self.dc = kwargs.get("dc", 512) # 不同模型不一样
        self.dnope = kwargs.get("dnope", kwargs.get("qk_nope_head_dim", 0))
        self.drope = kwargs.get("drope", kwargs.get("qk_rope_head_dim", 0))
        self.kv_cache_per_layer = kwargs.get("kv_cache_per_layer", 0)
        self.attention_per_layer = kwargs.get("attention_per_layer", 0)
        self.linear_per_layer = kwargs.get("linear_per_layer", 0)
        self.total = kwargs.get("total", 0)
        self.flops_per_token = kwargs.get("flops_per_token", 0)

    def calculate_kv_cache_per_layer(self, attention_type: str) -> float:
        if attention_type == "MHA" or attention_type == "GQA":
            kv_cache = 2 * self.nk * self.dh * self.N
        elif attention_type == "MLA":
            kv_cache = (self.dc + self.drope) * self.N
        elif attention_type == "GVA":
            kv_cache = (self.H + self.nk * self.dh) * self.N
        elif attention_type == "GHA":
            kv_cache = (self.nk * self.dh + self.nv * self.dh) * self.N
        elif attention_type == "GTA":
            kv_cache = (self.nk * self.dh + self.nc * self.dl) * self.N
        else:
            raise ValueError(f"未知注意力类型: {attention_type}")
        return kv_cache

    def calculate_computation_per_layer(self, attention_type: str) -> tuple:
        if attention_type == "MHA":
            # MHA: Attention=(2*nh*dh*N^2), Linear=(4*N*H^2)
            attention_flops = 2 * self.nh * self.dh * (self.N ** 2)
            linear_flops = 4 * self.N * (self.H ** 2)
        elif attention_type == "GQA":
            # GQA: Attention=(2*nh*dh*N^2), Linear=(2*N*H^2 + 2*nk*dh*N*H)
            attention_flops = 2 * self.nh * self.dh * (self.N ** 2)
            linear_flops = 2 * self.N * (self.H ** 2) + 2 * self.nk * self.dh * self.N * self.H
        elif attention_type == "MLA":
            # MLA: Attention=(nh(drope+2*dnope)*N^2), Linear=((dc+drope)*H + nh*(drope+dnope)*H + 2*nh*dl*dnope + H^2)*N)
            attention_flops = self.nh * (self.drope + 2*self.dnope) * (self.N**2)
            linear_flops = ((self.dc + self.drope) * self.H + self.nh * (self.drope + self.dnope) * self.H + 2 * self.nh * self.dl * self.dnope + self.H ** 2) * self.N
        elif attention_type == "GVA":
            # GVA: Attention=((nq*dh + nk*dh)*N^2), Linear=(2*N*H^2 + 2*nk*dh*N*H)
            attention_flops = (self.nq * self.dh + self.nk * self.dh) * (self.N ** 2)
            linear_flops = 2 * self.N * (self.H ** 2) + 2 * self.nk * self.dh * self.N * self.H
        elif attention_type == "GHA":
            # GHA: Attention=((nq*dh + nh*dh)*N^2), Linear=(N*H^2 + nq*dh*N*H + nk*dh*N*H + nv*dh*N*H)
            attention_flops = (self.nq * self.dh + self.nh * self.dh) * (self.N ** 2)
            linear_flops = self.N * (self.H ** 2) + self.nq * self.dh * self.N
        elif attention_type == "GTA":
            # GTA: Attention=(nq*(dk+dl)*N^2), Linear=(2*N*H^2 + (nq*dh + nk*dh + nc*dl + dl)*N*H)
            attention_flops = 2 * self.nk * self.dh * (self.N ** 2)
            linear_flops = 2 * self.N * (self.H ** 2) + (self.nq * self.dh + self.nk * self.dh + self.nc * self.dl + self.dl) * self.N * self.H
        else:
            raise ValueError(f"未知注意力类型: {attention_type}")
        return attention_flops, linear_flops

    def cal_flops(self) -> dict:
        """
        一个主方法，用于计算给定注意力类型的KV缓存和总计算量。
        """
        kv_cache = self.calculate_kv_cache_per_layer(self.attention_type)
        attention_flops, linear_flops = self.calculate_computation_per_layer(self.attention_type)
        self.kv_cache_per_layer = kv_cache
        self.attention_per_layer = attention_flops
        self.linear_per_layer = linear_flops
        self.total = self.num_hidden_layers * (kv_cache + attention_flops + linear_flops)
        self.flops_per_token = (self.num_hidden_layers * (kv_cache + attention_flops + linear_flops)) / self.N
        return {
            "attention_type": self.attention_type,
            "kv_cache_per_layer": self.kv_cache_per_layer,
            "attention_per_layer": self.attention_per_layer,
            "linear_per_layer": self.linear_per_layer,
            "total": self.total,
            "flops_per_token": self.flops_per_token,
        }

    def cal_total_flops(self) -> float:
        return self.cal_flops().get("total")

    def cal_flops_per_token(self) -> float:
        return self.cal_flops().get("flops_per_token")

    def estimate_flops_per_token(self) -> float:
        """
        使用公式估算FLOPs per token。
        公式为: 2 * num_hidden_layers * hidden_size * hidden_size
        """
        return 2 * self.num_hidden_layers * self.H * self.H


if __name__ == "__main__":
    # Qwen2.5-1.5B
    common_params = {
        # - Number of Attention Heads (GQA): 12 for Q and 2 for KV
        "attention_type": "GQA",
        "num_hidden_layers": 28,
        "p_tokens": 128, # Test n-prompt
        "n_tokens": 512, # Test n-gen
        "hidden_size": 1536, # "hidden_size": 1536
        # number of Attention Heads (GQA): 12 for Q and 2 for KV
        # nq = 12 nk = nv = 1
        "num_attention_heads": 12, # "num_attention_heads": 12
        "num_key_value_heads": 2, # "num_key_value_heads": 2
    }
    #
    # # PLM-1.8B-Instruct
    # common_params = {
    #     "attention_type": "MLA",
    #     "num_hidden_layers": 32,
    #     "p_tokens": 128, # Test n-prompt
    #     "n_tokens": 512,  # Test n-gen
    #     "hidden_size": 2048, # "hidden_size": 2048
    #     "num_attention_heads": 16, # "num_attention_heads": 16
    #     "num_key_value_heads": 16, # "num_key_value_heads": 16
    #     "qk_nope_head_dim": 128, # "qk_nope_head_dim": 128,
    #     "qk_rope_head_dim": 64, # "qk_rope_head_dim": 64,
    # }
    #
    # # SmolLM2-1.7B-Instruct
    # common_params = {
    #     "p_tokens": 128, # Test n-prompt
    #     "n_tokens": 512,  # Test n-gen
    #     "attention_type": "MHA",
    #     "hidden_size": 2048,
    #     "num_attention_heads": 32,
    #     "num_hidden_layers": 16,
    #     "num_key_value_heads": 32,
    # }
    print(common_params)
    calculator = AttentionFlopsCalculator()
    calculator.set_params(**common_params)
    flops_dict = calculator.cal_flops()
    print(flops_dict)

    flops_per_token = calculator.cal_flops_per_token()
    print(f"根据表格计算所得:{flops_per_token} flops per token")

    num_hidden_layers = 16
    hidden_size = 2048
    flops_per_token = calculator.estimate_flops_per_token()
    print(f"使用公式估算所得:{flops_per_token} flops per token")
